# why_are_govt_websites_so_bad

Thirty-three government service pages from eleven countries, screenshotted, embedded with CLIP, and clustered with k-means to see what visual families they fall into.

**Read the write-up:** https://rogersinghchugh.github.io/why_are_govt_websites_so_bad/

It explains every step of the pipeline with small worked examples (embeddings, tile averaging, L2 normalisation, k-means, silhouette, t-SNE) and records what the clusters show. The same observations are in [`results/observations.md`](results/observations.md).

## What is here

| Path | Purpose |
|---|---|
| `resources/websites.xlsx` | The 33 URLs: 11 countries × tax identifier, tax e-filing, passport |
| `take_screenshots.py` | Playwright capture with banner and modal dismissal |
| `utils.py` | The popup handler (generic multilingual pass plus a few site-specific steps) |
| `embed_cluster.py` | CLIP or ResNet101 embeddings, k-means or Ward, silhouette, plots |
| `results/` | Cluster plots, thumbnail maps, dendrogram, CSVs, `observations.md` |
| `docs/` | The GitHub Pages site |
| `data_analysis.ipynb` | The earlier raw-pixel clustimage attempt, kept for reference |

## Run it

```
uv sync
uv run playwright install chromium
uv run python take_screenshots.py            # 33 shots -> resources/pictures/
uv run python embed_cluster.py               # k by silhouette -> results/
uv run python embed_cluster.py --k 5         # five-cluster reading aid
uv run python embed_cluster.py --backbone resnet101 --out results/resnet101
```

Screenshots are not committed; the first command above regenerates them. The first embedding run downloads the CLIP weights.

Method inspired by [sabrinas.space](https://sabrinas.space/).
