"""Cluster portal screenshots on deep image embeddings instead of raw pixels.

Usage:
    python embed_cluster.py                           # resources/pictures -> results/  (CLIP)
    python embed_cluster.py --backbone resnet101      # Sabrina's setup: ImageNet ResNet101 features
    python embed_cluster.py --pictures resources/pictures_prev --out results/prev
    python embed_cluster.py --k 4                     # force the number of clusters
    python embed_cluster.py --no-tiles                # whole-image embedding only

Each screenshot is embedded as [whole image (squashed to 224x224)] + [mean of a 2x3 tile grid],
L2-normalised, then clustered with Ward agglomerative clustering. k is chosen by silhouette
unless --k is given. Outputs: clusters.csv, embeddings.npy, tsne_map.png (thumbnails coloured by
cluster), dendrogram.png (leaves coloured by country), cluster_sheet.png, summary.txt.
"""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

logging.getLogger("transformers").setLevel(logging.ERROR)

BACKBONES = {
    "clip": "openai/clip-vit-base-patch32",
    "resnet101": "microsoft/resnet-101",
}
CAT_NAMES = {"1": "tax_id", "2": "efiling", "3": "passport"}
TILE_GRID = (2, 3)  # rows, cols


# ----------------------------------------------------------------------------- data
def parse_name(path: Path):
    country, n = path.stem.rsplit("_", 1)
    return country, CAT_NAMES.get(n, n)


def load_paths(folder: Path):
    paths = sorted(p for p in folder.glob("*.png") if "_" in p.stem)
    if not paths:
        raise SystemExit(f"no *.png files in {folder}")
    return paths


# ----------------------------------------------------------------------- embedding
class Backbone:
    """Wraps an image model as: list of PIL images -> L2-normalised feature rows."""

    SIZE = 224

    def __init__(self, name: str, model_id: str | None = None):
        import transformers
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        transformers.utils.logging.set_verbosity_error()
        self.name = name
        self.model_id = model_id or BACKBONES[name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if name == "clip":
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self.proc = CLIPProcessor.from_pretrained(self.model_id).image_processor
            self.proc_kwargs = dict(do_resize=False, do_center_crop=False)
            # explicit vision tower + projection: get_image_features returns different types across versions
            self._forward = lambda pv: self.model.visual_projection(
                self.model.vision_model(pixel_values=pv).pooler_output
            )
        elif name == "resnet101":
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
            self.proc = AutoImageProcessor.from_pretrained(self.model_id)
            self.proc_kwargs = dict(do_resize=False)
            self._forward = lambda pv: self.model(pixel_values=pv).pooler_output.flatten(1)
        else:
            raise SystemExit(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")

    @torch.no_grad()
    def embed(self, pils):
        # resize ourselves (squash to square) so no part of the page is cropped away
        pils = [im.convert("RGB").resize((self.SIZE, self.SIZE), Image.BICUBIC) for im in pils]
        inputs = self.proc(images=pils, return_tensors="pt", **self.proc_kwargs)
        feats = self._forward(inputs["pixel_values"].to(self.device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


def tiles_of(im: Image.Image, grid=TILE_GRID):
    rows, cols = grid
    w, h = im.size
    tw, th = w // cols, h // rows
    return [im.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)) for r in range(rows) for c in range(cols)]


def embed_all(paths, backbone: Backbone, use_tiles=True):
    whole, tiled = [], []
    for p in paths:
        im = Image.open(p).convert("RGB")
        whole.append(backbone.embed([im])[0])
        if use_tiles:
            t = backbone.embed(tiles_of(im)).mean(axis=0)
            tiled.append(t / np.linalg.norm(t))
    X = np.stack(whole)
    if use_tiles:
        X = np.concatenate([X, np.stack(tiled)], axis=1)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


# ---------------------------------------------------------------------- clustering
def cluster(X, k, method="kmeans"):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(X)
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def silhouette_curve(X, method="kmeans", kmin=2, kmax=8):
    out = {}
    for k in range(kmin, min(kmax, len(X) - 1) + 1):
        out[k] = silhouette_score(X, cluster(X, k, method), metric="cosine")
    return out


def cohesion_table(X, groups):
    """Mean within-group cosine similarity vs mean similarity to everything else."""
    S = X @ X.T
    g = np.array(groups)
    rows = []
    for name in sorted(set(groups)):
        idx = np.where(g == name)[0]
        other = np.where(g != name)[0]
        if len(idx) < 2:
            continue
        within = S[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
        between = S[np.ix_(idx, other)].mean()
        rows.append((name, len(idx), within, between, within - between))
    cols = ["group", "n", "within_sim", "between_sim", "gap"]
    return pd.DataFrame(rows, columns=cols).sort_values("gap", ascending=False)


# ------------------------------------------------------------------------- plots
def plot_tsne_map(df, paths, out_path, backbone_name="clip", thumb_w=120):
    k = df["cluster"].nunique()
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(26, 16))
    for (_, r), p in zip(df.iterrows(), paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb_w, thumb_w))
        box = AnnotationBbox(
            OffsetImage(np.asarray(im), zoom=1.0),
            (r.tsne_x, r.tsne_y),
            frameon=True,
            bboxprops=dict(edgecolor=cmap(int(r.cluster) % 10), linewidth=3),
        )
        ax.add_artist(box)
        ax.annotate(
            f"{r.country} / {r.category}",
            (r.tsne_x, r.tsne_y),
            xytext=(0, -thumb_w * 0.36),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.scatter(df.tsne_x, df.tsne_y, s=0)
    pad = 0.08
    ax.set_xlim(df.tsne_x.min() - pad * np.ptp(df.tsne_x) - 1, df.tsne_x.max() + pad * np.ptp(df.tsne_x) + 1)
    ax.set_ylim(df.tsne_y.min() - pad * np.ptp(df.tsne_y) - 1, df.tsne_y.max() + pad * np.ptp(df.tsne_y) + 1)
    handles = [plt.Line2D([], [], color=cmap(c % 10), lw=4, label=f"cluster {c}") for c in range(k)]
    ax.legend(handles=handles, loc="upper left")
    ax.set_title(f"t-SNE of {backbone_name} embeddings; frame colour = cluster")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_cluster_scatter(df, out_path, method="kmeans", backbone_name="clip"):
    """Plain cluster plot: t-SNE projection, one colour per cluster, X marks the mean of each cluster's points."""
    k = df["cluster"].nunique()
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(15, 10))
    for c in range(k):
        sub = df[df.cluster == c]
        ax.scatter(sub.tsne_x, sub.tsne_y, s=150, color=cmap(c % 10), edgecolor="black", linewidth=0.6,
                   label=f"cluster {c}  (n={len(sub)})", zorder=3)
        ax.scatter(sub.tsne_x.mean(), sub.tsne_y.mean(), marker="X", s=420, color=cmap(c % 10),
                   edgecolor="black", linewidth=1.4, zorder=4)
    for r in df.itertuples():
        ax.annotate(f"{r.country.replace('_', ' ')} / {r.category}", (r.tsne_x, r.tsne_y),
                    xytext=(7, 5), textcoords="offset points", fontsize=8.5)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"{method} clusters on {backbone_name} embeddings, shown on a t-SNE projection (X = cluster mean)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_silhouette_curve(curve, chosen_k, out_path, method="kmeans"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ks, vals = list(curve), list(curve.values())
    ax.plot(ks, vals, marker="o")
    ax.axvline(chosen_k, color="red", linestyle="--", label=f"chosen k = {chosen_k}")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("silhouette (cosine)")
    ax.set_title(f"{method}: silhouette by k")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_dendrogram(X, df, out_path, backbone_name="clip"):
    Z = linkage(X, method="ward")
    countries = sorted(df.country.unique())
    cmap = plt.get_cmap("tab20")
    colour = {c: cmap(i % 20) for i, c in enumerate(countries)}
    fig, ax = plt.subplots(figsize=(18, 8))
    labels = [f"{c}/{k}" for c, k in zip(df.country, df.category)]
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=9, ax=ax)
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(colour[lbl.get_text().split("/")[0]])
    ax.set_title(f"Ward dendrogram on {backbone_name} embeddings; label colour = country")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_cluster_sheet(df, paths, out_path, tw=320, th=180, cols=8):
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        big = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = big = ImageFont.load_default()
    groups = [(c, df.index[df.cluster == c].tolist()) for c in sorted(df.cluster.unique())]
    height = sum(((len(ix) + cols - 1) // cols) * (th + 18) + 34 for _, ix in groups)
    canvas = Image.new("RGB", (tw * cols, height), "white")
    d = ImageDraw.Draw(canvas)
    y = 0
    for c, ix in groups:
        d.text((6, y + 6), f"CLUSTER {c}  (n={len(ix)})", fill="red", font=big)
        y += 34
        for j, i in enumerate(ix):
            im = Image.open(paths[i]).convert("RGB").resize((tw, th))
            x, yy = (j % cols) * tw, y + (j // cols) * (th + 18)
            canvas.paste(im, (x, yy))
            d.rectangle([x, yy, x + tw - 1, yy + th - 1], outline="black")
            d.text((x + 3, yy + th + 1), f"{df.country[i]} / {df.category[i]}", fill="black", font=font)
        y += ((len(ix) + cols - 1) // cols) * (th + 18)
    canvas.save(out_path)


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pictures", default="resources/pictures")
    ap.add_argument("--out", default="results")
    ap.add_argument("--backbone", choices=list(BACKBONES), default="clip")
    ap.add_argument("--model", default=None, help="override the Hugging Face model id for the backbone")
    ap.add_argument("--method", choices=["kmeans", "ward"], default="kmeans", help="clustering algorithm")
    ap.add_argument("--k", type=int, default=None, help="number of clusters (default: best silhouette in 2..8)")
    ap.add_argument("--no-tiles", action="store_true", help="embed the whole image only")
    args = ap.parse_args()

    pictures, out = Path(args.pictures), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = load_paths(pictures)
    meta = [parse_name(p) for p in paths]
    df = pd.DataFrame(
        {"file": [p.name for p in paths], "country": [m[0] for m in meta], "category": [m[1] for m in meta]}
    )

    tiles_desc = "off" if args.no_tiles else str(TILE_GRID)
    backbone = Backbone(args.backbone, args.model)
    print(f"embedding {len(paths)} images with {backbone.name} ({backbone.model_id}) on {backbone.device}, tiles={tiles_desc} ...")
    X = embed_all(paths, backbone, use_tiles=not args.no_tiles)
    np.save(out / "embeddings.npy", X)

    curve = silhouette_curve(X, args.method)
    k = args.k or max(curve, key=curve.get)
    df["cluster"] = cluster(X, k, args.method)
    df["silhouette"] = silhouette_samples(X, df.cluster, metric="cosine")
    xy = TSNE(
        n_components=2, perplexity=min(8, len(X) - 1), metric="cosine", init="pca", random_state=0
    ).fit_transform(X)
    df["tsne_x"], df["tsne_y"] = xy[:, 0], xy[:, 1]
    df.to_csv(out / "clusters.csv", index=False)

    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    lines = [f"backbone={backbone.name} ({backbone.model_id})  method={args.method}  tiles={tiles_desc}  dim={X.shape[1]}  n={len(X)}"]
    lines.append("silhouette by k: " + "  ".join(f"k={kk}:{v:.3f}" for kk, v in curve.items()))
    lines.append(f"chosen k={k}  silhouette={curve.get(k, float('nan')):.3f}")
    for c in sorted(df.cluster.unique()):
        members = df[df.cluster == c]
        lines.append(f"--- cluster {c} (n={len(members)}) ---")
        lines.append("  " + ", ".join(f"{r.country}/{r.category}" for r in members.itertuples()))
    lines.append("\ncluster x country:\n" + pd.crosstab(df.cluster, df.country).to_string())
    lines.append("\ncluster x category:\n" + pd.crosstab(df.cluster, df.category).to_string())
    lines.append("\ncountry cohesion (within-country similarity minus similarity to all other countries):")
    lines.append(cohesion_table(X, df.country.tolist()).to_string(index=False, float_format=f3))
    lines.append("\ncategory cohesion:")
    lines.append(cohesion_table(X, df.category.tolist()).to_string(index=False, float_format=f3))
    lines.append("\nlowest per-sample silhouette:")
    worst = df.nsmallest(6, "silhouette")[["file", "cluster", "silhouette"]]
    lines.append(worst.to_string(index=False, float_format=f3))
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    plot_cluster_scatter(df, out / "cluster_plot.png", args.method, backbone.name)
    plot_silhouette_curve(curve, k, out / "silhouette_by_k.png", args.method)
    plot_tsne_map(df, paths, out / "tsne_map.png", backbone.name)
    plot_dendrogram(X, df, out / "dendrogram.png", backbone.name)
    plot_cluster_sheet(df, paths, out / "cluster_sheet.png")
    print(f"\nwrote {out}/: clusters.csv, embeddings.npy, summary.txt, cluster_plot.png, silhouette_by_k.png, "
          "tsne_map.png, dendrogram.png, cluster_sheet.png")


if __name__ == "__main__":
    main()
