# Clustering observations

Dataset: 33 viewport screenshots (1920x1080) of government service portals, 11 countries x 3 services
(tax identifier, income-tax e-filing, passport application). Captured 2026-10-03 with cookie banners and
interstitials dismissed.

Method: CLIP ViT-B/32 image embedding (whole page + mean of a 2x3 tile grid, L2-normalised),
k-means (20 restarts), k chosen by cosine silhouette. t-SNE is used for display only.

## How many clusters

| k | silhouette |
|---|---|
| 2 | 0.264 |
| 3 | 0.128 |
| 4 | 0.164 |
| 5 | 0.124 |
| 8 | 0.164 |

Silhouette peaks at k = 2 and that split is "one small, visually unusual group vs everything else".
Beyond k = 2 the structure is gradual rather than sharp, so the five-cluster cut below is a reading aid,
not a claim that five distinct design families exist. Plots: `clip/cluster_plot.png` (k = 2),
`clip_k5/cluster_plot.png` and `clip_k5/tsne_map.png` (k = 5), `clip_k5/cluster_sheet.png` (members).

## The five style clusters (k = 5)

**Cluster 0, white text-first service pages (14 pages).** White background, a single brand-colour
header band, left-aligned heading and body text, blue links, one or two buttons. Photos are absent or
small. This is the dominant mode in the set. Pages with a notice carousel or a small illustration
still land here as long as text carries the first screen.

**Cluster 1, dense portals and banners (9 pages).** Busy first screens: a wide banner photo or
illustration of people or landmarks, grids of icon tiles, news lists, and several competing calls to
action. Strong colour fills and three or more distinct content blocks visible above the fold. This is
the "portal home page" look rather than a single-task page.

**Cluster 2, full-bleed scenic hero (4 pages).** One large landscape photograph fills most of the
viewport, with a login card or a short headline on top and very little text. Visually the closest to a
consumer travel or brand site. Three of the four are passport pages.

**Cluster 3, near-empty white (3 pages).** Most of the viewport is blank. A small header, then a form
box or a block of terms text. Single-task, legal and transactional in tone, with no imagery at all.

**Cluster 4, pastel panels, mascots and notices (3 pages).** Blocks of pastel background (mint, cyan,
cream), illustrated mascot characters, red-bordered notice boxes, carousels, and many small links.
This is the most distinct visual language in the set and is what the k = 2 split isolates.

## Cross-cutting observations

- **Service type does not determine style.** Cluster 0 contains tax-ID, e-filing, and passport pages
  in equal measure. The one category tendency is that scenic-hero pages are mostly passport pages.
- **Style is set by the publishing organisation, not the task.** Five of the eleven three-page sets
  land entirely in one cluster, and the hierarchical view (`clip/dendrogram.png`) shows the same:
  pages from one government merge with each other before merging with anything else.
- **The common case is plain.** Fourteen of 33 pages are white, text-led, and low-saturation. The
  "dark and empty" style that the inspiring blog found common on commercial sites is essentially
  absent from these government portals.
- **Imagery splits into two kinds.** Photographs are of landscapes or passport documents.
  Illustrations are of people or mascots. The two rarely appear on the same page.
- **Interstitials are common.** Roughly a quarter of the pages opened with a cookie banner, a notice
  modal, or both. One page needed four consecutive modals dismissed before any content was reachable.
  These overlays had dominated the earlier raw-pixel clustering and had to be removed to see style.
- **Read the 2-D map loosely.** k-means runs in the 1024-dimensional embedding space. One page is
  grouped with the small cluster at k = 2 yet sits far from it on the t-SNE map, so distances in the
  plot are suggestive, not literal.

## Comparison with ResNet101 features

Re-running with an ImageNet ResNet101 backbone (`resnet101/`), the features used in the inspiring
analysis, gives a weaker silhouette (0.21 at k = 4). The pastel-mascot group still co-clusters but is
mixed with nine other pages, and the ResNet features reward shared chrome (identical header bars)
more than overall page style. CLIP separates style more cleanly on this small sample.
