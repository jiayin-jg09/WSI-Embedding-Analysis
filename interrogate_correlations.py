"""
interrogate_correlations.py
===========================
Interrogate each dimension-omics correlation in the catalog BEYOND the single OS
survival flag. Five axes, all on CACHED data (UNI2 mean matrix + the omics
parquets + the 31-method npz) -- no H5, memory-light:

  1. Per-cancer consistency   (within-cancer rho per tumor type; generality)
  2. Other endpoints/pheno    (DSS/PFI/DFI survival + stage/grade/age/sex/smoking)
  3. Cross-omic coherence      (does a dim's biology corroborate across modalities)
  4. Statistical robustness    (bootstrap CI, purity-controlled rho, agg stability)
  5. Aberrant-phenotype battery(divergent/sign-flip, cancer-unique, discordant
                                tumors, bimodal dims, instability, heterogeneity,
                                multivariate outlier tumors)

Outputs (BOTH gitignored -- contain real names, never committed, never on site):
  - augments results_v2/dim_omics/mentor_decoder.xlsx (new cols + "Aberrant cases" sheet)
  - writes  results_v2/dim_omics/correlation_interrogation.pdf (data + figures)

Minimum-support rule: only correlations with >=100 observed cases are reported;
at the per-cancer level a cancer counts only with >=100 complete pairs (so CHOL,
n=39, never qualifies).

Run:  python interrogate_correlations.py
"""
import os
import gc
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from uni2_omics_correlation import OMICS_DIR_DEFAULT, OMIC_SPECS, load_omic
from uni2_tile_galleries import _hashed_code
from wsi_survival_pipeline import thermal_pause
from make_mentor_decoder import build_decoder, guide_lines, COLS as DEC_COLS

warnings.filterwarnings("ignore")

DIR = os.path.join("results_v2", "dim_omics")
CATALOG = os.path.join(DIR, "catalog.csv")
NPZ = os.path.join("results_v2", "agg_full", "full_methods.npz")
META = os.path.join("results_v2", "agg_full", "meta.csv")
OM = OMICS_DIR_DEFAULT
CLIN_F = os.path.join(OM, "CLINICAL_ENHANCED.parquet")
PURITY_F = os.path.join(OM, "PURITY_CONSOLIDATED.parquet")
VOL_F = os.path.join(OM, "COMPOSITE_VOLATILITY_SCORES.parquet")
XLSX = os.path.join(DIR, "mentor_decoder.xlsx")
PDF = os.path.join(DIR, "correlation_interrogation.pdf")

MIN_CASES = 100
CANCERS_5OMIC = ["immune", "immune_signatures", "rppa", "expression", "cnv"]
COOLDOWN = 0.5


# ---------------------------------------------------------------- small utils
def code_of(row):
    return _hashed_code(row["dim"], row["omic"], row.get("feature_name", ""))


def spearman(x, y, min_n=MIN_CASES):
    """Spearman rho over complete pairs; (nan, n) if fewer than min_n."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < min_n:
        return np.nan, n
    rx = stats.rankdata(x[m]); ry = stats.rankdata(y[m])
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (float((rx * ry).sum() / denom) if denom else np.nan), n


def p_from_rho(rho, n):
    if not np.isfinite(rho) or n < 4:
        return np.nan
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho * rho))
    return float(2 * stats.t.sf(abs(t), n - 2))


# ---------------------------------------------------------------- data loading
def load_blocks(methods_wanted):
    """Per-patient matrices for several npz aggregation blocks; npz touched ONCE."""
    d = np.load(NPZ, allow_pickle=True)
    methods = [str(m) for m in d["method_names"]]
    bd = int(d["block_dim"])
    feats = d["features"]                      # big, transient
    meta = pd.read_csv(META)
    tum = meta["sample_type"].values == "tumor"
    pids = meta["participant_id"].values[tum]
    uniq = pd.unique(pids)
    cols = [f"dim_{i}" for i in range(bd)]
    out = {}
    for m in methods_wanted:
        if m not in methods:
            continue
        blk = feats[:, methods.index(m) * bd:(methods.index(m) + 1) * bd][tum]
        X = np.vstack([blk[pids == p].mean(0) for p in uniq]).astype(np.float32)
        out[m] = pd.DataFrame(X, index=uniq, columns=cols)
        del blk, X
    del feats
    d.close(); gc.collect()
    print(f"loaded npz blocks {list(out)} | {len(uniq)} patients x {bd} dims", flush=True)
    return out


def load_features(cat, index):
    """Per-patient matrix of just the catalog feature columns, one omic at a time
    (column-filtered for the wide omics); aligned to `index`."""
    Y = pd.DataFrame(index=index)
    for omic in CANCERS_5OMIC:
        feats = cat.loc[cat["omic"] == omic, "feature"].unique().tolist()
        fn, prefix, _ = OMIC_SPECS[omic]
        path = os.path.join(OM, fn)
        if omic == "rppa":
            Yo = load_omic("rppa", OM)                      # 640 cols, renamed
        else:
            import pyarrow.parquet as pq
            schema = set(pq.ParquetFile(path).schema.names)
            read = [c for c in feats if c in schema]
            extra = ["participant_id"] if "participant_id" in schema else []
            Yo = pd.read_parquet(path, columns=read + extra)
            if "participant_id" in Yo.columns:
                Yo = Yo.set_index("participant_id")
            Yo = Yo[~Yo.index.duplicated(keep="first")].apply(pd.to_numeric, errors="coerce")
        Yo = Yo.reindex(index)
        take = [f for f in feats if f in Yo.columns]
        for c in take:
            Y[c] = Yo[c].astype(np.float32)
        print(f"  features[{omic}]: {len(take)}/{len(feats)} columns", flush=True)
        del Yo; gc.collect(); thermal_pause(COOLDOWN)
    return Y


def load_clinical(index):
    keep = ["project_id", "OS", "OS.time", "DSS", "DSS.time", "PFI", "PFI.time",
            "DFI", "DFI.time", "tumor_grade", "age_at_index", "gender",
            "tobacco_smoking_status"]
    clin = pd.read_parquet(CLIN_F)
    if clin.index.name != "participant_id" and "participant_id" in clin.columns:
        clin = clin.set_index("participant_id")
    clin = clin[~clin.index.duplicated(keep="first")]
    clin = clin[[c for c in keep if c in clin.columns]].reindex(index)
    cancer = clin["project_id"].astype(str).str.replace("TCGA-", "", regex=False)
    # purity (consensus, fallback cpe)
    pur = pd.read_parquet(PURITY_F)
    if pur.index.name != "participant_id" and "participant_id" in pur.columns:
        pur = pur.set_index("participant_id")
    pur = pur[~pur.index.duplicated(keep="first")]
    pcol = "purity_consensus" if "purity_consensus" in pur.columns else "purity_cpe"
    purity = pd.to_numeric(pur[pcol], errors="coerce").reindex(index)
    # genomic instability (composite_volatility) + tumor stage
    vol = pd.read_parquet(VOL_F)
    if vol.index.name != "participant_id" and "participant_id" in vol.columns:
        vol = vol.set_index("participant_id")
    vol = vol[~vol.index.duplicated(keep="first")]
    instab = pd.to_numeric(vol.get("composite_volatility"), errors="coerce").reindex(index) \
        if "composite_volatility" in vol.columns else pd.Series(np.nan, index=index)
    stage = vol.get("tumor_stage") if "tumor_stage" in vol.columns else None
    stage = stage.reindex(index) if stage is not None else pd.Series(np.nan, index=index)
    return clin, cancer, purity, instab, stage


# ---------------------------------------------------------------- axis 1 + 5a/5b
GRADE_MAP = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "Low Grade": 1.5, "High Grade": 3.5}
STAGE_MAP = {"stage i": 1, "stage ii": 2, "stage iii": 3, "stage iv": 4}


def axis1_and_divergence(cat, Xm, Y, cancer):
    """Per-cancer within-cancer rho for every link -> consistency + divergence."""
    Xv = Xm.values; dim_ix = {c: i for i, c in enumerate(Xm.columns)}
    feat_ix = {c: i for i, c in enumerate(Y.columns)}
    Yv = Y.values
    cancers = [c for c in pd.unique(cancer.dropna())
               if (cancer == c).sum() >= MIN_CASES]
    cmasks = {c: (cancer.values == c) for c in cancers}
    rows = []
    for _, r in cat.iterrows():
        di = dim_ix.get(r["dim"]); fi = feat_ix.get(r["feature"])
        rec = {"website_code": code_of(r), "n_cancers_qualified": 0,
               "n_cancers_concordant": 0, "rho_cancer_min": np.nan,
               "rho_cancer_max": np.nan, "rho_spread": np.nan,
               "divergent": False, "sign_flip": False, "unique_cancer": ""}
        if di is None or fi is None:
            rows.append(rec); continue
        x = Xv[:, di].astype(float); y = Yv[:, fi].astype(float)
        sign0 = np.sign(r["rho"])
        rhos, sig_cancers = {}, []
        for c in cancers:
            m = cmasks[c]
            rho, n = spearman(x[m], y[m])
            if np.isfinite(rho):
                rhos[c] = rho
                if p_from_rho(rho, n) < 0.05:
                    sig_cancers.append((c, rho))
        if rhos:
            vals = np.array(list(rhos.values()))
            rec["n_cancers_qualified"] = len(rhos)
            rec["n_cancers_concordant"] = sum(
                1 for c, rho in rhos.items()
                if np.sign(rho) == sign0 and p_from_rho(rho, int(cmasks[c].sum())) < 0.05)
            rec["rho_cancer_min"] = round(float(vals.min()), 3)
            rec["rho_cancer_max"] = round(float(vals.max()), 3)
            rec["rho_spread"] = round(float(vals.max() - vals.min()), 3)
            strong = [c for c, rho in rhos.items() if abs(rho) >= 0.2
                      and p_from_rho(rho, int(cmasks[c].sum())) < 0.05]
            weak = [c for c, rho in rhos.items() if abs(rho) < 0.1]
            rec["divergent"] = bool(strong and weak)
            rec["sign_flip"] = bool(any(rho >= 0.15 for rho in vals)
                                    and any(rho <= -0.15 for rho in vals))
            if len(sig_cancers) == 1 and len(rhos) >= 3:
                only = sig_cancers[0]
                if all(abs(rhos[c]) < 0.1 for c in rhos if c != only[0]):
                    rec["unique_cancer"] = only[0]
        rows.append(rec)
    return pd.DataFrame(rows), cancers


# ---------------------------------------------------------------- axis 2
def dim_survival(Xm, clin, dims, endpoint):
    """Stratified-by-cancer Cox HR/p per dim on `endpoint` (OS/DSS/PFI/DFI)."""
    from lifelines import CoxPHFitter
    ev, tt = endpoint, endpoint + ".time"
    if ev not in clin.columns or tt not in clin.columns:
        return {}
    base = pd.DataFrame({"evt": pd.to_numeric(clin[ev], errors="coerce"),
                         "dur": pd.to_numeric(clin[tt], errors="coerce"),
                         "strata": clin["project_id"].astype(str)}, index=clin.index)
    ok = base["evt"].notna() & (base["dur"] > 0)
    base = base[ok]
    Xs = Xm.reindex(base.index)
    out = {}
    for j, dim in enumerate(dims):
        v = Xs[dim].values.astype(float)
        sd = v.std() or 1.0
        df = base.copy()
        df["z"] = (v - np.nanmean(v)) / sd
        df = df.dropna(subset=["z"])
        try:
            cph = CoxPHFitter().fit(df, "dur", "evt", strata=["strata"])
            out[dim] = (round(float(np.exp(cph.params_["z"])), 3),
                        float(cph.summary.loc["z", "p"]))
        except Exception:
            out[dim] = (np.nan, np.nan)
        if j % 80 == 79:
            gc.collect(); thermal_pause(COOLDOWN)
    return out


def axis2_endpoints_pheno(cat, Xm, clin, cancer, stage):
    dims = sorted(cat["dim"].unique())
    perdim = {d: {} for d in dims}
    for ep in ["DSS", "PFI", "DFI"]:
        res = dim_survival(Xm, clin, dims, ep)
        for d in dims:
            hr, p = res.get(d, (np.nan, np.nan))
            perdim[d][f"{ep.lower()}_hr"] = hr
            perdim[d][f"{ep.lower()}_sig"] = bool(np.isfinite(p) and p < 0.05)
        print(f"  survival[{ep}] done", flush=True)
        gc.collect(); thermal_pause(COOLDOWN)

    # within-cancer phenotype associations (pooled rank, cancer-residualized)
    grade = clin["tumor_grade"].map(lambda g: GRADE_MAP.get(str(g).strip(), np.nan)) \
        if "tumor_grade" in clin.columns else pd.Series(np.nan, index=Xm.index)
    stg = stage.astype(str).str.lower().str.strip().map(
        lambda s: next((v for k, v in STAGE_MAP.items() if s.startswith(k)), np.nan))
    age = pd.to_numeric(clin.get("age_at_index"), errors="coerce")
    sex = clin.get("gender", pd.Series(index=Xm.index)).astype(str).str.lower()
    smoke = clin.get("tobacco_smoking_status", pd.Series(index=Xm.index)).astype(str)
    cv = cancer.values

    def within_spearman(dimvals, ph):
        # residualize both on cancer dummies, then Spearman (within-cancer assoc)
        m = np.isfinite(dimvals) & np.isfinite(ph)
        if m.sum() < MIN_CASES:
            return np.nan, np.nan
        d = pd.get_dummies(pd.Series(cv[m])).to_numpy(float)
        rx = stats.rankdata(dimvals[m]); ry = stats.rankdata(ph[m])
        bx = rx - d @ np.linalg.lstsq(d, rx, rcond=None)[0]
        by = ry - d @ np.linalg.lstsq(d, ry, rcond=None)[0]
        rho, n = spearman(bx, by)
        return rho, p_from_rho(rho, n)

    for d in dims:
        dv = Xm[d].values.astype(float)
        gr_r, gr_p = within_spearman(dv, grade.values.astype(float))
        st_r, st_p = within_spearman(dv, stg.values.astype(float))
        ag_r, ag_p = within_spearman(dv, age.values.astype(float))
        # sex: point-biserial via 0/1
        sx = np.where(sex.values == "male", 1.0, np.where(sex.values == "female", 0.0, np.nan))
        sx_r, sx_p = within_spearman(dv, sx)
        perdim[d].update({"grade_rho": _r(gr_r), "grade_p": _r(gr_p),
                          "stage_rho": _r(st_r), "stage_p": _r(st_p),
                          "age_rho": _r(ag_r), "age_p": _r(ag_p),
                          "sex_p": _r(sx_p)})
    out = pd.DataFrame([{"dim": d, **perdim[d]} for d in dims])
    return out


def _r(v, k=4):
    return round(float(v), k) if v is not None and np.isfinite(v) else np.nan


# ---------------------------------------------------------------- axis 4
def axis4_robustness(cat, blocks, Y, cancer, purity):
    Xm = blocks["mean"]
    Xv = Xm.values; dim_ix = {c: i for i, c in enumerate(Xm.columns)}
    feat_ix = {c: i for i, c in enumerate(Y.columns)}
    Yv = Y.values
    cdum = pd.get_dummies(cancer.fillna("NA")).to_numpy(float)
    pv = purity.values.astype(float)
    pv = np.where(np.isfinite(pv), pv, np.nanmean(pv))
    rng = np.random.default_rng(42)
    alt = [m for m in ["median", "percentile_75", "std"] if m in blocks]
    rows = []
    for k, (_, r) in enumerate(cat.iterrows()):
        di = dim_ix.get(r["dim"]); fi = feat_ix.get(r["feature"])
        rec = {"website_code": code_of(r), "rho_recomp": np.nan, "rho_ci_low": np.nan,
               "rho_ci_high": np.nan, "rho_purity_ctrl": np.nan, "agg_stable": np.nan}
        if di is not None and fi is not None:
            x = Xv[:, di].astype(float); y = Yv[:, fi].astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            n = int(m.sum())
            if n >= MIN_CASES:
                rx = stats.rankdata(x[m]); ry = stats.rankdata(y[m])
                # residualize ranks on cancer dummies -> WITHIN-CANCER rho (matches catalog)
                Dc = cdum[m]
                crx = rx - Dc @ np.linalg.lstsq(Dc, rx, rcond=None)[0]
                cry = ry - Dc @ np.linalg.lstsq(Dc, ry, rcond=None)[0]

                def _corr(a, b):
                    a = a - a.mean(); b = b - b.mean()
                    d = np.sqrt((a * a).sum() * (b * b).sum())
                    return float((a * b).sum() / d) if d else np.nan
                rec["rho_recomp"] = _r(_corr(crx, cry), 3)
                # bootstrap CI of the within-cancer rho (vectorized over B for this link)
                B = 1000
                idx = rng.integers(0, n, size=(B, n))
                a = crx[idx]; b = cry[idx]
                a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
                num = (a * b).sum(1)
                den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
                bc = num / np.where(den == 0, np.nan, den)
                rec["rho_ci_low"] = _r(np.nanpercentile(bc, 2.5), 3)
                rec["rho_ci_high"] = _r(np.nanpercentile(bc, 97.5), 3)
                del idx, a, b
                # purity-controlled: residualize on cancer + purity
                D = np.column_stack([Dc, (pv[m] - pv[m].mean())])
                brx = rx - D @ np.linalg.lstsq(D, rx, rcond=None)[0]
                bry = ry - D @ np.linalg.lstsq(D, ry, rcond=None)[0]
                pr, pn = spearman(brx, bry)
                rec["rho_purity_ctrl"] = _r(pr, 3)
                # aggregation stability: sign + nominal sig under median/p75/std
                sign0 = np.sign(r["rho"]); stable = 0
                for mth in alt:
                    xa = blocks[mth].values[:, di].astype(float)
                    ra, na = spearman(xa, y)
                    if np.isfinite(ra) and np.sign(ra) == sign0 and p_from_rho(ra, na) < 0.05:
                        stable += 1
                rec["agg_stable"] = f"{stable}/{len(alt)}"
        rows.append(rec)
        if k % 100 == 99:
            gc.collect(); thermal_pause(COOLDOWN)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- axis 3
def axis3_coherence(cat, Y, Xm, cancer):
    """Lightweight per-dim cross-omic coherence: does a dim have concordant-sign
    significant hits in >=2 modalities for the SAME gene/protein? Uses the catalog
    rows we already have (top links) + name harmonization where available."""
    # map ENSEMBL->HUGO and RPPA protein->gene
    try:
        em = pd.read_parquet(os.path.join(OM, "ENSEMBL_HUGO_MAPPING_V2.parquet"))
        ens2sym = {}
        cols = [c for c in em.columns]
        if "ensembl" in cols and "symbol" in cols:
            ens2sym = dict(zip(em["ensembl"].astype(str), em["symbol"].astype(str)))
    except Exception:
        ens2sym = {}

    def gene_of(omic, feat, fname):
        f = str(fname)
        if omic == "expression":
            e = f.replace("expr_", "").split(".")[0]
            return ens2sym.get(e, e).upper()
        if omic == "cnv":
            return f.replace("cnv_", "").split(".")[0].upper()
        if omic == "rppa":
            return str(feat).replace("rppa_", "").upper()
        return None

    cat = cat.copy()
    cat["gene"] = [gene_of(o, fe, fn) for o, fe, fn in
                   zip(cat["omic"], cat["feature"], cat["feature_name"])]
    perdim = {}
    for dim, g in cat.groupby("dim"):
        genes = g.dropna(subset=["gene"])
        score = 0; omics_hit = set()
        # protein<->transcript and expr<->cnv: same gene present in >=2 omics, same sign
        for gene, gg in genes.groupby("gene"):
            if gene and gg["omic"].nunique() >= 2:
                signs = np.sign(gg["rho"].values)
                if len(set(signs[signs != 0])) == 1:           # concordant sign
                    score += 1
                    omics_hit.update(gg["omic"].unique())
        # program-backed (has a Hallmark label that isn't 'none')
        prog = g["program"].dropna().astype(str)
        prog_backed = int(any(p.lower() not in ("none", "nan", "") for p in prog))
        perdim[dim] = {"coherence_score": int(min(3, score + prog_backed)),
                       "coherent_omics": ",".join(sorted(omics_hit))}
    out = pd.DataFrame([{"dim": d, **v} for d, v in perdim.items()])
    return out


# ---------------------------------------------------------------- axis 5 c-g
def axis5_aberrant(cat, blocks, Y, cancer, instab):
    Xm = blocks["mean"]
    dim_ix = {c: i for i, c in enumerate(Xm.columns)}
    feat_ix = {c: i for i, c in enumerate(Y.columns)}
    Xv = Xm.values; Yv = Y.values
    cancers = [c for c in pd.unique(cancer.dropna()) if (cancer == c).sum() >= MIN_CASES]

    # (d) bimodality (Sarle) + (e) instability + (f) heterogeneity, per dim
    std_b = blocks.get("std"); iqr_b = blocks.get("iqr")
    perdim = {}
    for dim in sorted(cat["dim"].unique()):
        di = dim_ix[dim]; x = Xv[:, di].astype(float)
        bimodal_in = []
        for c in cancers:
            v = x[cancer.values == c]; v = v[np.isfinite(v)]
            if len(v) >= MIN_CASES:
                s = stats.skew(v); k = stats.kurtosis(v, fisher=False)
                bc = (s * s + 1) / (k + 3 * (len(v) - 1) ** 2 / ((len(v) - 2) * (len(v) - 3)))
                if bc > 0.555:
                    bimodal_in.append(c)
        ins_r, ins_n = spearman(x, instab.values.astype(float))
        het = np.nan
        if std_b is not None:
            sdv = std_b.values[:, di].astype(float)
            mn = np.abs(x) + 1e-6
            het = float(np.nanmedian(sdv / mn))
        perdim[dim] = {"bimodal_in": ",".join(bimodal_in),
                       "instability_rho": _r(ins_r, 3),
                       "hetero_score": _r(het, 4)}
    perdim_df = pd.DataFrame([{"dim": d, **v} for d, v in perdim.items()])

    # (c) discordant tumors: residual outliers from dim~feature fit, top links only
    aberrant = []
    top = cat.reindex(cat["rho"].abs().sort_values(ascending=False).index).head(60)
    for _, r in top.iterrows():
        di = dim_ix.get(r["dim"]); fi = feat_ix.get(r["feature"])
        if di is None or fi is None:
            continue
        x = Xv[:, di].astype(float); y = Yv[:, fi].astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < MIN_CASES:
            continue
        rx = stats.rankdata(x[m]); ry = stats.rankdata(y[m])
        rx = (rx - rx.mean()) / (rx.std() or 1); ry = (ry - ry.mean()) / (ry.std() or 1)
        resid = rx - r["rho"] * ry              # deviation from the expected relation
        idx = np.where(m)[0]
        zr = (resid - resid.mean()) / (resid.std() or 1)
        for j in np.where(np.abs(zr) > 3)[0]:
            aberrant.append({"barcode": Xm.index[idx[j]], "kind": "discordant",
                             "code": code_of(r), "cancer": cancer.iloc[idx[j]],
                             "residual_z": round(float(zr[j]), 2),
                             "side": "dim-high/feat-low" if zr[j] > 0 else "dim-low/feat-high"})

    # (g) multivariate outlier tumors: Mahalanobis within cancer on top PCs
    for c in cancers:
        mask = cancer.values == c
        Xc = Xv[mask]
        Xc = Xc[:, np.isfinite(Xc).all(0)]
        if Xc.shape[0] < MIN_CASES:
            continue
        Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
        # top-20 PCs
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        K = min(20, S.shape[0])
        scores = U[:, :K] * S[:K]
        d2 = (scores ** 2 / (scores.var(0) + 1e-9)).sum(1)
        thr = np.percentile(d2, 99)
        bars = Xm.index[mask]
        for j in np.where(d2 >= thr)[0]:
            aberrant.append({"barcode": bars[j], "kind": "multivar_outlier",
                             "code": "", "cancer": c,
                             "residual_z": round(float(np.sqrt(d2[j])), 2),
                             "side": "global"})
    aberrant_df = pd.DataFrame(aberrant)
    return perdim_df, aberrant_df, cancers


# ---------------------------------------------------------------- PDF report
def write_pdf(dec, aberrant_df, cancers):
    with PdfPages(PDF) as pdf:
        # cover
        fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
        fig.text(0.5, 0.9, "Dimension-omics correlations: interrogation report",
                 ha="center", fontsize=15, fontweight="bold")
        intro = ("Five axes beyond the OS survival flag, on the UNI2 pan-cancer cohort.\n"
                 "Minimum support: >=100 cases (per-cancer: >=100 complete pairs; CHOL excluded).\n"
                 "PRIVATE -- contains real feature names. Pair with mentor_decoder.xlsx.\n\n"
                 f"Links analysed: {len(dec)}   Cancers qualifying: {', '.join(cancers)}")
        fig.text(0.1, 0.78, intro, fontsize=10, va="top")
        pdf.savefig(fig); plt.close(fig)

        def hist(ax, data, title, xlabel, bins=30, color="#0d9488"):
            data = np.asarray(data, float); data = data[np.isfinite(data)]
            ax.hist(data, bins=bins, color=color)
            ax.set_title(title, fontsize=10, fontweight="bold"); ax.set_xlabel(xlabel)

        # axis1 + 5a/5b
        fig, ax = plt.subplots(2, 2, figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        if "n_cancers_concordant" in dec:
            hist(ax[0, 0], dec["n_cancers_concordant"], "Per-cancer consistency",
                 "# cancers concordant (same sign, p<0.05)")
        if "rho_spread" in dec:
            hist(ax[0, 1], dec["rho_spread"], "Cancer divergence (Axis 5a)",
                 "rho spread (max-min across cancers)", color="#c2410c")
        if "divergent" in dec:
            vc = dec["divergent"].value_counts()
            ax[1, 0].bar([str(i) for i in vc.index], vc.values, color="#c2410c")
            ax[1, 0].set_title("Divergent links", fontsize=10, fontweight="bold")
        if "unique_cancer" in dec:
            u = dec.loc[dec["unique_cancer"] != "", "unique_cancer"].value_counts()
            if len(u):
                ax[1, 1].bar(u.index, u.values, color="#0d9488")
            ax[1, 1].set_title("Cancer-unique links (Axis 5b)", fontsize=10, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # axis2 endpoints/pheno
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5)); fig.patch.set_facecolor("white")
        eps = [c for c in ["dss_sig", "pfi_sig", "dfi_sig"] if c in dec]
        if eps:
            vals = [int(dec[c].fillna(False).astype(bool).sum()) for c in eps]
            ax[0].bar([e.split("_")[0].upper() for e in eps], vals, color="#0d9488")
            ax[0].set_title("Links significant per endpoint", fontsize=10, fontweight="bold")
        phe = [c for c in ["grade_p", "stage_p", "age_p", "sex_p"] if c in dec]
        if phe:
            vals = [int((pd.to_numeric(dec[c], errors="coerce") < 0.05).sum()) for c in phe]
            ax[1].bar([c.split("_")[0] for c in phe], vals, color="#5eead4")
            ax[1].set_title("Links significant per phenotype", fontsize=10, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # axis3 coherence + axis4 robustness
        fig, ax = plt.subplots(2, 2, figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        if "coherence_score" in dec:
            vc = dec["coherence_score"].value_counts().sort_index()
            ax[0, 0].bar(vc.index.astype(str), vc.values, color="#0d9488")
            ax[0, 0].set_title("Cross-omic coherence score", fontsize=10, fontweight="bold")
        if {"rho_recomp", "rho_ci_low", "rho_ci_high"}.issubset(dec.columns):
            t = dec.dropna(subset=["rho_ci_low", "rho_recomp"]).reindex(
                dec["rho"].abs().sort_values(ascending=False).index).head(25)
            cen = pd.to_numeric(t["rho_recomp"], errors="coerce").values
            lo = np.clip(cen - pd.to_numeric(t["rho_ci_low"], errors="coerce").values, 0, None)
            hi = np.clip(pd.to_numeric(t["rho_ci_high"], errors="coerce").values - cen, 0, None)
            ax[0, 1].errorbar(cen, np.arange(len(t)), xerr=[lo, hi],
                              fmt="o", color="#0d9488", ms=3, lw=0.8)
            ax[0, 1].set_title("Within-cancer rho, bootstrap 95% CI (top links)",
                               fontsize=9, fontweight="bold")
            ax[0, 1].set_yticks([])
        if {"rho_recomp", "rho_purity_ctrl"}.issubset(dec.columns):
            ax[1, 0].scatter(pd.to_numeric(dec["rho_recomp"], errors="coerce"),
                             pd.to_numeric(dec["rho_purity_ctrl"], errors="coerce"),
                             s=6, alpha=0.4, color="#0d9488")
            ax[1, 0].plot([-1, 1], [-1, 1], "--", color="#475569", lw=0.8)
            ax[1, 0].set_xlabel("within-cancer rho"); ax[1, 0].set_ylabel("purity-controlled rho")
            ax[1, 0].set_title("Purity robustness", fontsize=10, fontweight="bold")
        if "hetero_score" in dec:
            hist(ax[1, 1], pd.to_numeric(dec["hetero_score"], errors="coerce"),
                 "Intratumoral heterogeneity (5f)", "hetero score", color="#c2410c")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # aberrant cases summary
        fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
        fig.text(0.5, 0.93, "Aberrant cases (Axis 5c/5g)", ha="center",
                 fontsize=13, fontweight="bold")
        if len(aberrant_df):
            txt = aberrant_df["kind"].value_counts().to_string()
            by_c = aberrant_df["cancer"].value_counts().to_string()
            fig.text(0.1, 0.85, "By kind:\n" + txt + "\n\nBy cancer:\n" + by_c,
                     fontsize=10, va="top", family="monospace")
        else:
            fig.text(0.1, 0.85, "none flagged", fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    print(f"wrote {PDF}", flush=True)


# ---------------------------------------------------------------- main
def main():
    cat = pd.read_csv(CATALOG)
    blocks = load_blocks(["mean", "median", "percentile_75", "std", "iqr"])
    Xm = blocks["mean"]
    Y = load_features(cat, Xm.index)
    clin, cancer, purity, instab, stage = load_clinical(Xm.index)

    per_link = pd.DataFrame({"website_code": [code_of(r) for _, r in cat.iterrows()]})
    per_dim_frames = []

    def safe(label, fn):
        try:
            print(f"[axis] {label} ...", flush=True)
            return fn()
        except Exception as e:
            print(f"  !! {label} failed: {e}", flush=True)
            return None

    a1, cancers = safe("1 per-cancer + divergence",
                       lambda: axis1_and_divergence(cat, Xm, Y, cancer)) or (None, [])
    if a1 is not None:
        per_link = per_link.merge(a1, on="website_code", how="left")
    a2 = safe("2 endpoints/phenotypes",
              lambda: axis2_endpoints_pheno(cat, Xm, clin, cancer, stage))
    if a2 is not None:
        per_dim_frames.append(a2)
    a3 = safe("3 cross-omic coherence", lambda: axis3_coherence(cat, Y, Xm, cancer))
    if a3 is not None:
        per_dim_frames.append(a3)
    a4 = safe("4 robustness", lambda: axis4_robustness(cat, blocks, Y, cancer, purity))
    if a4 is not None:
        per_link = per_link.merge(a4, on="website_code", how="left")
    a5 = safe("5 aberrant battery",
              lambda: axis5_aberrant(cat, blocks, Y, cancer, instab))
    aberrant_df = pd.DataFrame()
    if a5 is not None:
        a5d, aberrant_df, _ = a5
        per_dim_frames.append(a5d)

    # assemble: decoder + per-link + per-dim
    dec = build_decoder()
    dec = dec.merge(per_link, on="website_code", how="left")
    cat_codes = cat.assign(website_code=[code_of(r) for _, r in cat.iterrows()])[
        ["website_code", "dim"]]
    dec = dec.merge(cat_codes.drop_duplicates("website_code"), on="website_code", how="left",
                    suffixes=("", "_c"))
    dec["dim_key"] = dec["dim"].where(dec["dim"].notna(), dec.get("dim_c"))
    for f in per_dim_frames:
        dec = dec.merge(f, left_on="dim_key", right_on="dim", how="left",
                        suffixes=("", "_y"))
        dec = dec.drop(columns=[c for c in dec.columns if c.endswith("_y") or c == "dim_c"],
                       errors="ignore")
    dec = dec.drop(columns=["dim_key"], errors="ignore")

    # write workbook (Decoder + How to read + Aberrant cases)
    extra_guide = [
        "", "INTERROGATION COLUMNS (added beyond the base decoder)",
        "  n_cancers_qualified / n_cancers_concordant  per-cancer consistency (>=100 cases each).",
        "  rho_cancer_min/max, rho_spread              spread of within-cancer rho across cancers.",
        "  divergent / sign_flip / unique_cancer       aberrant (Axis 5a/5b): context-specific links.",
        "  dss_/pfi_/dfi_hr & _sig                      stratified Cox on other survival endpoints.",
        "  grade_/stage_/age_/sex_ rho/p               within-cancer phenotype associations.",
        "  coherence_score / coherent_omics            cross-omic corroboration (0-3).",
        "  rho_recomp                                  rho recomputed within-cancer (matches the CI).",
        "  rho_ci_low/high                             bootstrap 95% CI on the within-cancer rho.",
        "  rho_purity_ctrl                             rho after controlling tumor purity.",
        "  agg_stable                                  holds under median/P75/std aggregations (k/3).",
        "  bimodal_in / instability_rho / hetero_score aberrant phenotype signals (Axis 5d/5e/5f).",
        "  See the 'Aberrant cases' sheet for individual flagged tumors (Axis 5c/5g).",
    ]
    guide = pd.DataFrame({"How to read this workbook": guide_lines() + extra_guide})
    with pd.ExcelWriter(XLSX, engine="openpyxl") as xl:
        dec.to_excel(xl, sheet_name="Decoder", index=False)
        guide.to_excel(xl, sheet_name="How to read", index=False)
        (aberrant_df if len(aberrant_df) else pd.DataFrame(
            columns=["barcode", "kind", "code", "cancer", "residual_z", "side"])
         ).to_excel(xl, sheet_name="Aberrant cases", index=False)
        xl.sheets["Decoder"].freeze_panes = "A2"
        xl.sheets["How to read"].column_dimensions["A"].width = 100
    print(f"wrote {XLSX}: {dec.shape[0]} rows x {dec.shape[1]} cols | "
          f"{len(aberrant_df)} aberrant cases", flush=True)

    write_pdf(dec, aberrant_df, cancers)


if __name__ == "__main__":
    main()
