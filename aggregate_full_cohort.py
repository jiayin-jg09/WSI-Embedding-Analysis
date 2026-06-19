"""
aggregate_full_cohort.py
========================
Thermal-safe re-aggregation of ALL 8-cancer slides from the H5 patch files,
computing the full menu of aggregation methods (median, geometric mean, IQR,
entropy, …) so they can be visualized by cancer type — not just the 6 "rich"
statistics cached in results_v2/aggregated_embeddings.csv.

Reads each H5 once (the only heavy step), computes ~31 aggregations in that
single pass, and checkpoints every 50 slides so an overheat/crash never loses
more than a minute of work (resume with --resume).

Designed for the thermally-constrained laptop:
  * loads one H5 at a time, chunked (reuses the tested pipeline loader)
  * explicit cooldown pause between files (default 2.0 s, --cooldown to change)
  * single-threaded numpy; aggressive gc

Run (recommended, in the background):
  python aggregate_full_cohort.py --resume --cooldown 2.0
Smoke test first:
  python aggregate_full_cohort.py --limit 3

Output: results_v2/agg_full/full_methods.npz  (features, method_names, block_dim)
        results_v2/agg_full/meta.csv          (filename, participant_id, sample_type)
"""
import os, sys, gc, json, argparse, time
import numpy as np

from wsi_survival_pipeline import (
    discover_h5_files, parse_tcga_barcode, load_h5_patches, thermal_pause,
)
from scipy import stats as sstats

H5_DIRS = ["TCGA UNI2 embeddings", "embeddings"]
OUT_DIR = os.path.join("results_v2", "agg_full")
CKPT = os.path.join(OUT_DIR, ".agg_full_ckpt.json")
PARTIAL = os.path.join(OUT_DIR, ".agg_full_partial.npy")
EPS = 1e-8
MAX_PATCHES = 5000   # subsample very large slides; order-stats stay ~identical


def _safe(v):
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def methods_for(p):
    """Compute all aggregations for one slide's patch matrix p (N x D).

    Fast path: subsample to <= MAX_PATCHES, then sort ONCE along the patch axis
    and read every order-statistic off the sorted array (percentiles, median,
    min/max, top-k, trimmed, IQR) instead of re-sorting per statistic.
    """
    n0 = p.shape[0]
    if n0 > MAX_PATCHES:
        idx = np.random.default_rng(n0).choice(n0, MAX_PATCHES, replace=False)
        p = p[idx]
    n = p.shape[0]
    s = np.sort(p, axis=0)                      # single O(N log N) sort

    def q(perc):
        return s[min(int(round(perc / 100.0 * (n - 1))), n - 1)]

    def topk(k, top=True):
        k = min(k, n)
        return s[n - k:].mean(0) if top else s[:k].mean(0)

    mean = p.mean(0); std = p.std(0); var = p.var(0)
    med = q(50); p5, p25, p75, p95 = q(5), q(25), q(75), q(95)
    smin, smax = s[0], s[-1]
    absp = np.abs(p)
    lo = int(0.1 * n)
    out = {
        "mean": mean, "median": med, "std": std, "var": var,
        "min": smin, "max": smax, "range": smax - smin,
        "mid_range": (smax + smin) / 2.0,
        "percentile_5": p5, "percentile_10": q(10), "percentile_25": p25,
        "percentile_75": p75, "percentile_90": q(90), "percentile_95": p95,
        "iqr": p75 - p25,
        "mad": np.median(np.abs(p - med), 0),
        "cv": std / (np.abs(mean) + EPS),
        "rms": np.sqrt((p ** 2).mean(0)),
        "sum_abs": absp.sum(0),
        "skewness": sstats.skew(p, axis=0),
        "kurtosis": sstats.kurtosis(p, axis=0),
        "geometric_mean": np.exp(np.log(absp + EPS).mean(0)),
        "harmonic_mean": n / (1.0 / (absp + EPS)).sum(0),
        "trimmed_mean": s[lo:n - lo].mean(0) if n - lo > lo else mean,
        "winsorized_mean": np.clip(p, p5, p95).mean(0),
        "top5_mean": topk(5, True), "top10_mean": topk(10, True),
        "bottom5_mean": topk(5, False), "bottom10_mean": topk(10, False),
        "entropy": 0.5 * np.log(2 * np.pi * np.e * var + EPS),
        "variance_weighted_mean": (p * (p.var(1) / (p.var(1).sum() + EPS))[:, None]).sum(0),
    }
    return {k: _safe(v) for k, v in out.items()}


METHOD_NAMES = list(methods_for(np.zeros((2, 4), dtype=np.float32)).keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="process only first N (smoke test)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    files = discover_h5_files(H5_DIRS)
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} H5 files; {len(METHOD_NAMES)} methods; "
          f"cooldown={args.cooldown}s", flush=True)

    rows = []            # each: concatenated method vector for one slide
    pids, stypes, fnames, completed = [], [], [], set()

    if args.resume and os.path.exists(CKPT) and os.path.exists(PARTIAL):
        ck = json.load(open(CKPT))
        completed = set(ck["completed"]); pids = ck["pids"]
        stypes = ck["stypes"]; fnames = ck["fnames"]
        rows = [r for r in np.load(PARTIAL)]
        print(f"  resuming: {len(completed)} already done", flush=True)

    t0 = time.time(); newly = 0; skipped = 0
    for i, path in enumerate(files):
        base = os.path.basename(path)
        if base in completed:
            continue
        try:
            pid, stype = parse_tcga_barcode(path)
            patches = load_h5_patches(path)
            vec = np.concatenate([methods_for(patches)[m] for m in METHOD_NAMES])
            del patches; gc.collect()
            rows.append(vec.astype(np.float32))
            pids.append(pid); stypes.append(stype); fnames.append(base)
            completed.add(base); newly += 1
            if newly % 10 == 0:
                rate = (time.time() - t0) / newly
                print(f"  [{len(completed)}/{len(files)}] {base}  "
                      f"({rate:.1f}s/slide, ~{rate*(len(files)-len(completed))/60:.0f} min left)",
                      flush=True)
            if newly % args.checkpoint_every == 0:
                np.save(PARTIAL, np.vstack(rows))
                json.dump({"completed": list(completed), "pids": pids,
                           "stypes": stypes, "fnames": fnames}, open(CKPT, "w"))
            thermal_pause(args.cooldown)
        except Exception as e:
            skipped += 1
            print(f"  SKIP {base}: {e}", flush=True)

    feats = np.vstack(rows).astype(np.float32)
    dim = feats.shape[1] // len(METHOD_NAMES)
    np.savez_compressed(os.path.join(OUT_DIR, "full_methods.npz"),
                        features=feats, method_names=np.array(METHOD_NAMES), block_dim=dim)
    import pandas as pd
    pd.DataFrame({"filename": fnames, "participant_id": pids,
                  "sample_type": stypes}).to_csv(os.path.join(OUT_DIR, "meta.csv"), index=False)
    for f in (CKPT, PARTIAL):
        if os.path.exists(f):
            try: os.remove(f)
            except OSError: pass
    print(f"\nDONE: {len(feats)} slides, {len(METHOD_NAMES)} methods x {dim} dims, "
          f"{skipped} skipped. Wrote {OUT_DIR}/full_methods.npz", flush=True)


if __name__ == "__main__":
    main()
