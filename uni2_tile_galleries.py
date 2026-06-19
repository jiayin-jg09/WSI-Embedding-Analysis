"""
uni2_tile_galleries.py
======================
Show the histology behind each dimension-omics link.

For each top distinct dimension d (from the catalog), take the top 10 cases by the
slide-level value of dim d, and from EACH case crop its 100 most-activating tiles
(the patches with the largest dim-d value). The 1000 tiles are laid out as one
montage per dimension: ten 10x10 per-case blocks, so each block is one case and
you see both within-case consistency and cross-case spread. Keyed by the catalog
CODE, so the molecular feature stays hidden.

Pipeline (thermal- and storage-safe):
  * case ranking uses the cached full_methods.npz mean block + meta.csv (no H5)
  * for each (dim, case) we read just that H5 column + coords, take the top 100
    patches, and crop them from the .svs (openslide), downsampling in memory
  * WSIs are read straight from --wsi-dir and never copied; only the montages
    (one JPEG per dimension) land in figures/galleries/

Run:
  python uni2_tile_galleries.py --wsi-dir "D:/TCGA WSI" --limit 1     # prototype
  python uni2_tile_galleries.py --wsi-dir "D:/TCGA WSI"               # top 100 dims
  python uni2_tile_galleries.py --write-web                           # rebuild page only
"""
import os
import re
import gc
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from PIL import Image

from wsi_survival_pipeline import thermal_pause

DIR = os.path.join("results_v2", "dim_omics")
CATALOG = os.path.join(DIR, "catalog.csv")
NPZ = os.path.join("results_v2", "agg_full", "full_methods.npz")
META = os.path.join("results_v2", "agg_full", "meta.csv")
H5_DIRS = ["TCGA UNI2 embeddings", "embeddings"]
OUT = os.path.join("figures", "galleries")
MANIFEST = os.path.join(DIR, "galleries_manifest.csv")
PATCH_LEVEL0 = 512          # patch size in level-0 pixels (from H5 attrs)
BG = (18, 22, 20)


def dim_num(dstr):
    return int(re.sub(r"\D", "", str(dstr)))


def h5_path(filename):
    for d in H5_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


def index_svs(wsi_dir):
    """Map TCGA barcode (and full stem) -> .svs path, walking the drive once."""
    idx = {}
    for p in glob.glob(os.path.join(wsi_dir, "**", "*.svs"), recursive=True):
        stem = os.path.splitext(os.path.basename(p))[0]
        idx.setdefault(stem, p)
        idx.setdefault(stem.split(".")[0], p)        # barcode = before first dot
    return idx


def match_svs(h5_filename, svs_idx):
    stem = h5_filename[:-3] if h5_filename.endswith(".h5") else h5_filename
    return svs_idx.get(stem) or svs_idx.get(stem.split(".")[0])


def case_block(tiles, grid, tile_px):
    """One case's tiles in a grid x grid block (in activation order)."""
    blk = Image.new("RGB", (grid * tile_px, grid * tile_px), BG)
    for i, t in enumerate(tiles[:grid * grid]):
        r, c = divmod(i, grid)
        blk.paste(t, (c * tile_px, r * tile_px))
    return blk


def assemble(blocks, block_cols, grid, tile_px, gap):
    """Arrange per-case blocks into one montage (blocks separated by a gap)."""
    bpx = grid * tile_px
    rows = int(np.ceil(len(blocks) / block_cols))
    W = block_cols * bpx + (block_cols + 1) * gap
    H = rows * bpx + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), BG)
    for i, blk in enumerate(blocks):
        r, c = divmod(i, block_cols)
        canvas.paste(blk, (gap + c * (bpx + gap), gap + r * (bpx + gap)))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi-dir", help="folder of .svs WSIs (external drive)")
    ap.add_argument("--limit", type=int, default=0, help="N top dimensions (0 = top 100)")
    ap.add_argument("--top-cases", type=int, default=10)
    ap.add_argument("--tiles-per-case", type=int, default=100)
    ap.add_argument("--block-grid", type=int, default=10, help="tiles per side in a case block")
    ap.add_argument("--block-cols", type=int, default=5, help="case blocks per montage row")
    ap.add_argument("--tile-px", type=int, default=40)
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--jpeg-quality", type=int, default=70)
    ap.add_argument("--cooldown", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="rebuild montages that already exist")
    ap.add_argument("--write-web", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.write_web:
        write_gallery_page(); return
    if not args.wsi_dir:
        sys.exit("--wsi-dir is required (path to the .svs files on the drive)")

    import h5py
    import openslide

    cat = pd.read_csv(CATALOG)
    cat["absrho"] = cat["rho"].abs()
    cat["dimn"] = cat["dim"].map(dim_num)
    rep = (cat.sort_values("absrho", ascending=False)
              .drop_duplicates("dimn", keep="first")
              .head(args.limit if args.limit else 100).reset_index(drop=True))
    dims = list(rep["dimn"])
    print(f"rendering {len(dims)} distinct-dimension galleries "
          f"({args.top_cases} cases x {args.tiles_per_case} tiles each)", flush=True)

    # case ranking from the cached mean block (no H5 needed)
    npz = np.load(NPZ, allow_pickle=True)
    methods = [str(m) for m in npz["method_names"]]; bd = int(npz["block_dim"])
    mb = npz["features"][:, methods.index("mean") * bd:(methods.index("mean") + 1) * bd]
    meta = pd.read_csv(META); tum = meta["sample_type"].values == "tumor"
    mb = mb[tum]; fnames = meta["filename"].values[tum]

    print(f"indexing .svs under {args.wsi_dir} ...", flush=True)
    svs_idx = index_svs(args.wsi_dir)
    print(f"  {len(svs_idx)} .svs keys", flush=True)

    montage_of = {}
    n_missing = 0
    for di, d in enumerate(dims):
        fn = f"dim_{d:04d}.jpg"
        if not args.force and os.path.exists(os.path.join(OUT, fn)):
            montage_of[d] = fn
            print(f"  [{di+1}/{len(dims)}] dim {d}: exists, skip", flush=True)
            continue
        cases = [fnames[i] for i in np.argsort(mb[:, d])[::-1][:args.top_cases]]
        blocks = []
        for s in cases:
            hp = h5_path(s); sp = match_svs(s, svs_idx)
            if hp is None or sp is None:
                n_missing += 1; continue
            try:
                with h5py.File(hp, "r") as h:
                    col = h["features"][0, :, d]
                    coords = h["coords"][0]
                k = min(args.tiles_per_case, col.shape[0])
                top = np.argsort(col)[::-1][:k]
                slide = openslide.OpenSlide(sp)
                tiles = []
                for idx in top:
                    x, y = int(coords[idx, 0]), int(coords[idx, 1])
                    tiles.append(slide.read_region((x, y), 0, (PATCH_LEVEL0, PATCH_LEVEL0))
                                 .convert("RGB").resize((args.tile_px, args.tile_px)))
                slide.close()
                blocks.append(case_block(tiles, args.block_grid, args.tile_px))
                del tiles, coords, col
            except Exception as e:
                print(f"    {s[:28]} fail: {e}", flush=True)
            gc.collect()
        if blocks:
            m = assemble(blocks, args.block_cols, args.block_grid, args.tile_px, args.gap)
            fn = f"dim_{d:04d}.jpg"
            m.save(os.path.join(OUT, fn), quality=args.jpeg_quality)
            montage_of[d] = fn
            print(f"  [{di+1}/{len(dims)}] dim {d}: {fn} ({len(blocks)} cases)", flush=True)
        thermal_pause(args.cooldown)

    print(f"\n{len(montage_of)} montages | {n_missing} cases skipped (missing)", flush=True)
    write_gallery_page()


# ----------------------------------------------------------------- website
def _hashed_code(dim, omic, feat_readable):
    """Opaque code (feature name hashed) -- never expose the feature on the web."""
    import hashlib
    dd = re.sub(r"\D", "", str(dim)) or "0"
    ab = {"expression": "EXPR", "cnv": "CNV", "immune_signatures": "SIG",
          "rppa": "RPPA", "immune": "IMM", "mutations": "MUT"}.get(omic, "X")
    h = hashlib.sha1(str(feat_readable).upper().encode()).hexdigest()[:5].upper()
    return f"U2-D{int(dd):04d}-{ab}-{h}"


def write_gallery_page():
    """Rebuild the manifest + page from catalog.csv and the montages on disk, with
    HASHED codes (so a stale feature-revealing code can never reach the site)."""
    if not os.path.exists(CATALOG):
        print("no catalog.csv; nothing to do"); return
    cat = pd.read_csv(CATALOG)
    cat["absrho"] = cat["rho"].abs(); cat["dimn"] = cat["dim"].map(dim_num)
    rep = (cat.sort_values("absrho", ascending=False)
              .drop_duplicates("dimn", keep="first").head(100))
    rows = []
    for _, r in rep.iterrows():
        fn = f"dim_{int(r['dimn']):04d}.jpg"
        if not os.path.exists(os.path.join(OUT, fn)):
            continue
        rows.append({"code": _hashed_code(r["dim"], r["omic"], r.get("feature_name", "")),
                     "rho": r["rho"], "n_models": r.get("n_models", ""),
                     "surv_hr": r.get("surv_hr", ""), "surv_sig": r.get("surv_sig", ""),
                     "montage": fn})
    man = pd.DataFrame(rows)
    man.to_csv(MANIFEST, index=False)
    cards = []
    for _, r in man.iterrows():
        surv = ""
        try:
            if str(r["surv_sig"]).lower() == "true":
                surv = f' &middot; surv {float(r["surv_hr"]):.2f}*'
        except Exception:
            pass
        cards.append(
            f'<figure class="gal-card">'
            f'<img src="../figures/galleries/{r["montage"]}" loading="lazy" '
            f'alt="tiles for {r["code"]}" data-zoom>'
            f'<figcaption><code>{r["code"]}</code><br>'
            f'<span class="muted">rho {float(r["rho"]):+.2f} &middot; '
            f'{r["n_models"]}/6 models{surv}</span></figcaption></figure>')
    grid = "\n      ".join(cards)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tile galleries · WSI Pan-Cancer</title>
<meta name="description" content="For each top dimension, a montage of the most-activating H&amp;E tiles: ten 10x10 blocks, one per top case, 100 tiles each.">
<link rel="stylesheet" href="../assets/styles.css">
<script>
  try {{ if (localStorage.getItem("wsi-theme") === "light")
    document.documentElement.setAttribute("data-theme", "light"); }} catch (e) {{}}
</script>
<script defer src="../assets/app.js"></script>
</head>
<body>

<nav>
  <div class="nav-inner">
    <a href="../index.html" class="brand">WSI · Pan-Cancer</a>
    <div class="nav-links">
      <a href="../index.html">Home</a>
      <a href="background.html">Background</a>
      <a href="overview.html">Story</a>
      <a href="methods.html">Methods</a>
      <div class="nav-dropdown">
        <button class="nav-drop-toggle" aria-haspopup="true" aria-expanded="false">Models ▾</button>
        <div class="nav-drop-menu">
          <a href="models-survival.html">Survival</a>
          <a href="models-classification.html">Classification</a>
          <a href="models-aggregation.html">Aggregation</a>
          <a href="models-clustering.html">Clustering &amp; Statistics</a>
          <a href="models-metrics.html">Metrics</a>
          <a href="models-omics.html">Biology</a>
          <a href="models-catalog.html">Catalog</a>
          <a href="models-tiles.html">Tiles</a>
        </div>
      </div>
      <a href="results.html">Results</a>
      <a href="figures.html">Figures</a>
      <button class="theme-toggle" aria-label="Toggle light / dark theme" title="Toggle theme">
        <span class="icon-light">🌙</span><span class="icon-dark">☀️</span>
      </button>
    </div>
  </div>
</nav>

<main id="top">
  <section class="hero">
    <div class="eyebrow">Models · tiles</div>
    <h1>What each dimension looks at</h1>
    <p class="subtitle">
      Each montage is one dimension from the <a href="models-catalog.html">catalog</a>. We take
      the top 10 cases by the dimension's slide-level value and, from each, the 100 most-activating
      H&amp;E tiles (20x, 256&nbsp;µm patches). The montage shows ten 10&times;10 blocks, one per
      case, so you can read within-case consistency (across a block) and cross-case spread (across
      blocks). Cards are labeled by code only; the molecular feature is not shown. Click to enlarge.
    </p>
  </section>
  <section id="galleries">
    <div class="gallery">
      {grid}
    </div>
  </section>
</main>

<footer>
  Built with the <span style="color: var(--accent);">Pine</span> theme ·
  TCGA Pan-Cancer WSI Embedding Analysis
</footer>

</body>
</html>
"""
    page = os.path.join("pages", "models-tiles.html")
    with open(page, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {page} ({len(man)} cards)", flush=True)


if __name__ == "__main__":
    main()
