"""
uni2_catalog.py
===============
A coded discovery catalog of the strongest UNI2 dimension <-> molecule links.

Honest framing: the *raw* dimension-omics correlations already exist in the lab's
data for SIX foundation models (ctranspath, kaiko-s8, lunit-dino, pathdino, sp22m,
uni2), pooled. What this catalog adds is the de-confounded view plus curation:
  * associations are WITHIN-CANCER (cancer-type regressed out)
  * each gets a CROSS-MODEL robustness count (in how many of the 6 models the same
    feature is strongly encoded) -> model-independent biology is the strong signal
  * each dimension gets a SURVIVAL flag (stratified Cox on overall survival)
  * each gets a stable opaque code, e.g. U2-D0953-EXPR-7A3F2 (feature name hashed)
  * grouped into modules (hub dimensions)
Mutations are kept as a separate (expectedly weak) section.

Outputs (results_v2/dim_omics/, gitignored): catalog.csv, catalog_mutations.csv,
modules_dims.csv, modules_programs.csv, and catalog_section.html (web fragment).
Run: python uni2_catalog.py
"""
import os
import gc
import re
import sys
import hashlib
import warnings
import numpy as np
import pandas as pd

from wsi_survival_pipeline import thermal_pause
from uni2_omics_correlation import OMICS_DIR_DEFAULT, load_uni2_patient_matrix
from uni2_vs_ctranspath import norm_key, best_per_feature

warnings.filterwarnings("ignore")

DIR = os.path.join("results_v2", "dim_omics")
CTP = os.path.join("..", "TCGA multiomics", "dim_omic_correlations")
MAPF = os.path.join(OMICS_DIR_DEFAULT, "ENSEMBL_HUGO_MAPPING_V2.parquet")
RPPA_MAP_F = os.path.join(OMICS_DIR_DEFAULT, "ARCHIVE", "RPPA_numeric_column_mapping.csv")
LABELS = os.path.join(DIR, "dimension_labels.csv")
MODELS = ["ctranspath", "kaiko-s8", "lunit-dino", "pathdino", "sp22m", "uni2"]
OMICS = ["expression", "cnv", "immune_signatures", "rppa", "immune"]   # mutations handled separately
ABBR = {"expression": "EXPR", "cnv": "CNV", "immune_signatures": "SIG",
        "rppa": "RPPA", "immune": "IMM", "mutations": "MUT"}
PER_OMIC = 200           # strongest associations kept per omic (5 omics -> ~1000)


def load_maps():
    m = pd.read_parquet(MAPF)
    m = m[m["query_success"]] if "query_success" in m.columns else m
    ens2hugo = dict(zip(m["original_column"], m["hugo_symbol"].astype(str)))
    rppa_map = {}
    if os.path.exists(RPPA_MAP_F):
        r = pd.read_csv(RPPA_MAP_F)
        rppa_map = {row["rppa_column"]: "rppa_" + str(row["protein_name"]).lstrip("X")
                    for _, row in r.iterrows()}
    return ens2hugo, rppa_map


def readable(feature, omic, ens2hugo):
    if omic == "expression":
        return ens2hugo.get(feature, feature.replace("expr_", ""))
    return re.sub(r"^(cnv_|immune_|sig_|rppa_|bin_|mut_)", "", feature)


def code(model_tag, dim, omic, feat_readable):
    """Code WITHOUT the feature name: the last block is an opaque, stable hash of
    the feature (same feature -> same hash), so the molecular identity is hidden
    on the web while the real names stay only in the local catalog.csv."""
    d = re.sub(r"\D", "", dim) or "0"
    h = hashlib.sha1(str(feat_readable).upper().encode()).hexdigest()[:5].upper()
    return f"{model_tag}-D{int(d):04d}-{ABBR[omic]}-{h}"


def load_assoc(omic, mode="residualized"):
    f = os.path.join(DIR, f"uni2_mean_{omic}.parquet")
    d = pd.read_parquet(f, columns=["dim", "feature", "rho", "q_value", "n", "mode"])
    d = d[(d["mode"] == mode) & (d["q_value"] < 0.05)].copy()
    d["omic"] = omic
    return d


def cross_model_counts(cat, rppa_map, q=0.80):
    """For each (omic, feature) in cat, count models (of 6) that rank the feature
    among their TOP 20% best-encoded features (a discriminating, relative test)."""
    enc, thr = {}, {}   # (omic, model) -> Series key->max|rho|, and that model's q80
    for omic in cat["omic"].unique():
        for model in MODELS:
            p = os.path.join(CTP, f"{model}_mean_{omic}.parquet")
            if os.path.exists(p):
                s = best_per_feature(p, omic, rppa_map, pooled_only=False)
                enc[(omic, model)] = s
                thr[(omic, model)] = float(s.quantile(q))
        gc.collect(); thermal_pause(0.5)
    counts = []
    for _, r in cat.iterrows():
        k = norm_key(r["feature"], r["omic"], rppa_map)
        c = 0
        for m in MODELS:
            s = enc.get((r["omic"], m))
            if s is not None and k in s.index and s[k] >= thr[(r["omic"], m)]:
                c += 1
        counts.append(c)
    return counts


def survival_by_dim(dims):
    """Stratified-by-cancer Cox HR + p of each dimension on overall survival."""
    from lifelines import CoxPHFitter
    X = load_uni2_patient_matrix("mean")
    clin = pd.read_parquet(os.path.join(OMICS_DIR_DEFAULT, "CLINICAL_ENHANCED.parquet"))
    if clin.index.name != "participant_id" and "participant_id" in clin.columns:
        clin = clin.set_index("participant_id")
    clin = clin[~clin.index.duplicated(keep="first")]
    common = X.index.intersection(clin.index)
    surv = clin.loc[common, ["OS", "OS.time", "project_id"]].apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.name != "project_id" else s)
    ok = surv["OS"].notna() & (surv["OS.time"] > 0)
    surv, Xc = surv[ok], X.loc[common][ok.values]
    out = {}
    for dim in dims:
        df = pd.DataFrame({"dur": surv["OS.time"].values, "evt": surv["OS"].astype(int).values,
                           "z": (Xc[dim].values - Xc[dim].mean()) / (Xc[dim].std() or 1),
                           "strata": surv["project_id"].values})
        try:
            cph = CoxPHFitter().fit(df, "dur", "evt", strata=["strata"])
            out[dim] = (float(np.exp(cph.params_["z"])), float(cph.summary.loc["z", "p"]))
        except Exception:
            out[dim] = (np.nan, np.nan)
    return out


def main():
    if "--web-only" in sys.argv:
        # rebuild the web table from the already-computed CSVs (no 6-model reload),
        # recomputing the hashed codes so the feature stays hidden
        cat = pd.read_csv(os.path.join(DIR, "catalog.csv"))
        cat["code"] = [code("U2", d, o, fn)
                       for d, o, fn in zip(cat["dim"], cat["omic"], cat["feature_name"])]
        hub = pd.read_csv(os.path.join(DIR, "modules_dims.csv"))
        pm = pd.read_csv(os.path.join(DIR, "modules_programs.csv"))
        mut = pd.read_csv(os.path.join(DIR, "catalog_mutations.csv"))
        mut["code"] = [code("U2", d, "mutations", fn)
                       for d, fn in zip(mut["dim"], mut["feature_name"])]
        write_web(cat, hub, pm, mut)
        return

    ens2hugo, rppa_map = load_maps()

    # ---- assemble within-cancer associations (non-mutation), BALANCED per omic ----
    # (immune signatures have much larger |rho| than expression/CNV, so a global
    #  top-N would be all signatures; take the strongest PER_OMIC from each instead)
    frames = [load_assoc(o) for o in OMICS]
    for f in frames:
        f["absrho"] = f["rho"].abs()

    def _select(omic, f):
        f = f.sort_values("absrho", ascending=False)
        # RPPA: one protein (BCL-XL) has the strongest within-cancer correlation across
        # nearly every dimension, so a plain top-N collapses the whole RPPA section to a
        # single protein. Keep the strongest dim PER protein so RPPA stays diverse.
        if omic == "rppa":
            f = f.drop_duplicates("feature")
        return f.head(PER_OMIC)
    cat = pd.concat([_select(o, f) for o, f in zip(OMICS, frames)],
                    ignore_index=True).sort_values("absrho", ascending=False).reset_index(drop=True)
    cat["feature_name"] = [readable(f, o, ens2hugo) for f, o in zip(cat["feature"], cat["omic"])]
    cat["code"] = [code("U2", d, o, fn)
                   for d, o, fn in zip(cat["dim"], cat["omic"], cat["feature_name"])]
    print(f"catalog: top {len(cat)} within-cancer associations "
          f"(|rho| {cat['absrho'].min():.2f}-{cat['absrho'].max():.2f})", flush=True)

    print("cross-model robustness over 6 models ...", flush=True)
    cat["n_models"] = cross_model_counts(cat, rppa_map)
    print("  omic dist:", cat["omic"].value_counts().to_dict(), flush=True)
    print("  n_models dist:", cat["n_models"].value_counts().sort_index().to_dict(), flush=True)

    print("survival association per dimension ...", flush=True)
    surv = survival_by_dim(sorted(cat["dim"].unique()))
    cat["surv_hr"] = [round(surv[d][0], 3) if not np.isnan(surv[d][0]) else np.nan for d in cat["dim"]]
    cat["surv_p"] = [surv[d][1] for d in cat["dim"]]
    cat["surv_sig"] = cat["surv_p"] < 0.05

    # programs (modules) from dimension labels
    prog = {}
    if os.path.exists(LABELS):
        lab = pd.read_csv(LABELS)
        prog = dict(zip(lab["dim"], lab["top_pathway"]))
    cat["program"] = cat["dim"].map(lambda d: prog.get(d, "none"))

    cols = ["code", "dim", "omic", "feature_name", "rho", "q_value", "n",
            "n_models", "surv_hr", "surv_p", "surv_sig", "program", "feature"]
    cat["rho"] = cat["rho"].round(3)
    cat[cols].to_csv(os.path.join(DIR, "catalog.csv"), index=False)
    print(f"wrote catalog.csv | {int(cat['surv_sig'].sum())} survival-linked | "
          f"{int((cat['n_models']>=5).sum())} robust in >=5/6 models", flush=True)

    # ---- modules ----
    hub = cat["dim"].value_counts().head(20).rename_axis("dim").reset_index(name="n_assoc")
    hub["program"] = hub["dim"].map(lambda d: prog.get(d, "none"))
    hub.to_csv(os.path.join(DIR, "modules_dims.csv"), index=False)
    pm = (cat[cat["program"] != "none"].groupby("program")
          .agg(n_assoc=("code", "size"), n_dims=("dim", "nunique")).reset_index()
          .sort_values("n_assoc", ascending=False))
    pm.to_csv(os.path.join(DIR, "modules_programs.csv"), index=False)

    # ---- mutation section (separate, expectedly weak) ----
    mut = load_assoc("mutations")
    mut = mut.reindex(mut["rho"].abs().sort_values(ascending=False).index).head(25)
    mut["feature_name"] = [readable(f, "mutations", ens2hugo) for f in mut["feature"]]
    mut["code"] = [code("U2", d, "mutations", fn)
                   for d, fn in zip(mut["dim"], mut["feature_name"])]
    mut["n_models"] = cross_model_counts(mut.assign(omic="mutations"), rppa_map)
    mut["rho"] = mut["rho"].round(3)
    mut[["code", "dim", "feature_name", "rho", "q_value", "n", "n_models"]].to_csv(
        os.path.join(DIR, "catalog_mutations.csv"), index=False)
    print(f"wrote catalog_mutations.csv (top {len(mut)} mutation links)", flush=True)

    write_web(cat, hub, pm, mut)


def _rows(df, cells):
    out = []
    for _, r in df.iterrows():
        tds = "".join(cells(r))
        out.append(f"<tr>{tds}</tr>")
    return "\n".join(out)


def write_web(cat, hub, pm, mut, per_omic=40):
    c = pd.concat([g.sort_values("rho", key=lambda s: s.abs(), ascending=False).head(per_omic)
                   for _, g in cat.groupby("omic")]).sort_values(
                   "rho", key=lambda s: s.abs(), ascending=False)
    top = len(c)
    assoc_rows = _rows(c, lambda r: [
        f'<td><code>{r["code"]}</code></td>',
        f'<td>{r["dim"]}</td>', f'<td>{ABBR[r["omic"]]}</td>',
        f'<td class="num" data-sort="{abs(r["rho"]):.3f}">{r["rho"]:+.3f}</td>',
        f'<td class="num" data-sort="{r["n_models"]}">{r["n_models"]}/6</td>',
        f'<td class="num" data-sort="{0 if pd.isna(r["surv_hr"]) else r["surv_hr"]}">'
        f'{"" if pd.isna(r["surv_hr"]) else ("%.2f%s" % (r["surv_hr"], "*" if r["surv_sig"] else ""))}</td>'])
    hub_rows = _rows(hub.head(12), lambda r: [
        f'<td>{r["dim"]}</td>', f'<td class="num">{r["n_assoc"]}</td>'])
    mut_rows = _rows(mut.head(15), lambda r: [
        f'<td><code>{r["code"]}</code></td>', f'<td>{r["dim"]}</td>',
        f'<td class="num" data-sort="{abs(r["rho"]):.3f}">{r["rho"]:+.3f}</td>',
        f'<td class="num">{r["n_models"]}/6</td>'])

    n_robust = int((cat["n_models"] >= 5).sum())
    n_surv = int(cat["surv_sig"].sum())
    html = f"""<!-- CATALOG:START (generated by uni2_catalog.py) -->
  <section id="catalog">
    <h2>A coded catalog of the strongest associations</h2>
    <p>
      A balanced selection of the strongest within-cancer links (~{per_omic} per data type) is
      shown here; click any column header to sort. Of the full {len(cat)}-row catalog,
      <strong>{n_robust}</strong> links are robust in at least 5 of 6 foundation models and
      <strong>{n_surv}</strong> involve a survival-associated dimension. The complete catalog,
      the module tables and the mutation list are written to
      <code>results_v2/dim_omics/</code>. Column meanings are explained above.
    </p>
    <table class="sortable">
      <thead><tr><th>Code</th><th>Dim</th><th>Omic</th>
        <th class="num">rho (within-cancer)</th><th class="num">models</th>
        <th class="num">surv HR/SD</th></tr></thead>
      <tbody>
{assoc_rows}
      </tbody>
    </table>

    <h3 style="color:var(--accent)">Hub dimensions</h3>
    <p>The dimensions that turn up most often in the catalog, i.e. that carry many strong
       associations at once.</p>
    <div class="fig-grid">
      <div>
        <table class="sortable"><thead><tr><th>Dim</th><th class="num">links</th></tr></thead>
        <tbody>
{hub_rows}
        </tbody></table></div>
    </div>

    <h3 style="color:var(--accent)">Mutation links (separate &mdash; expectedly weak)</h3>
    <p>Within cancer type the embedding barely tracks specific mutations; the strongest few are
       listed for completeness and mostly are not robust across models.</p>
    <table class="sortable">
      <thead><tr><th>Code</th><th>Dim</th><th class="num">rho</th><th class="num">models</th></tr></thead>
      <tbody>
{mut_rows}
      </tbody>
    </table>
  </section>
  <!-- CATALOG:END -->"""
    with open(os.path.join(DIR, "catalog_section.html"), "w", encoding="utf-8") as f:
        f.write(html)
    # inject into the Biology page between markers (idempotent)
    page = os.path.join("pages", "models-catalog.html")
    s = open(page, encoding="utf-8").read()
    if "<!-- CATALOG:START" in s:
        s = re.sub(r"<!-- CATALOG:START.*?<!-- CATALOG:END -->", html, s, flags=re.S)
    else:
        s = s.replace("<!-- CATALOG_PLACEHOLDER -->", html)
    open(page, "w", encoding="utf-8").write(s)
    print(f"injected catalog section into {page}", flush=True)


if __name__ == "__main__":
    main()
