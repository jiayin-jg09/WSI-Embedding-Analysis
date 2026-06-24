"""
make_mentor_decoder.py
======================
Build a private Excel "key" the mentor can use to read the coded
dimension-omics links shown on the website. The public site hides molecular
feature names behind hashed codes (e.g. U2-D1467-SIG-F1DF9); this workbook maps
every code back to its real dimension, omic, feature name and stats, plus a
plain-English guide.

It reuses the SAME hashing the website uses (`_hashed_code` from
uni2_tile_galleries) so the codes match the site exactly.

Output: results_v2/dim_omics/mentor_decoder.xlsx  (gitignored -- contains real
feature names; never committed and never on the website).

Run:  pip install openpyxl
      python make_mentor_decoder.py
"""
import os

import pandas as pd

from uni2_tile_galleries import _hashed_code

DIR = os.path.join("results_v2", "dim_omics")
CATALOG = os.path.join(DIR, "catalog.csv")
MAN = os.path.join(DIR, "galleries_manifest.csv")
MAN_OTHER = os.path.join(DIR, "galleries_manifest_other.csv")
OUT = os.path.join(DIR, "mentor_decoder.xlsx")

COLS = ["website_code", "dim", "omic", "feature_name", "program", "rho",
        "q_value", "n", "n_models", "surv_hr", "surv_p", "surv_sig",
        "on_tiles_page", "montage_base", "montage_beyond"]


def _montage_map(path, colname):
    """code -> montage filename from a galleries manifest (empty if absent)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["code", colname])
    m = pd.read_csv(path)[["code", "montage"]].rename(columns={"montage": colname})
    return m


def build_decoder():
    # drop the catalog's own (feature-revealing) code column; website_code replaces it
    cat = pd.read_csv(CATALOG).drop(columns=["code"], errors="ignore")
    cat["website_code"] = cat.apply(
        lambda r: _hashed_code(r["dim"], r["omic"], r.get("feature_name", "")), axis=1)

    base = _montage_map(MAN, "montage_base")
    other = _montage_map(MAN_OTHER, "montage_beyond")
    cat = cat.merge(base, left_on="website_code", right_on="code", how="left") \
             .drop(columns=["code"], errors="ignore")
    cat = cat.merge(other, left_on="website_code", right_on="code", how="left") \
             .drop(columns=["code"], errors="ignore")

    cat["on_tiles_page"] = cat["montage_base"].notna()
    cat["absrho"] = cat["rho"].abs()
    cat = cat.sort_values(["on_tiles_page", "absrho"], ascending=[False, False])
    return cat[COLS].reset_index(drop=True)


def guide_lines():
    return [
        "HOW TO READ THIS WORKBOOK",
        "",
        "WHAT THIS IS",
        "A private key to the coded dimension-molecule links shown on the project website.",
        "The public site shows only hashed codes (e.g. U2-D1467-SIG-F1DF9) so that molecular",
        "feature names never appear publicly. This file maps each code back to what it really is.",
        "",
        "KEEP THIS FILE PRIVATE. It contains real feature names. It is not on the website and is",
        "not committed to the code repository -- share it directly, not by posting it publicly.",
        "",
        "BACKGROUND",
        "UNI2 is a pathology foundation model that turns each image patch into a 1,536-number",
        "vector. A 'dimension' is one of those 1,536 learned features. We correlated each dimension",
        "against molecular measurements (per patient, within cancer type) to ask what biology each",
        "dimension tracks. The 'Decoder' sheet is the result, one row per dimension-feature link.",
        "",
        "CODE FORMAT:  U2-D{dim}-{OMIC}-{hash}",
        "  - {dim}  = the UNI2 dimension number (also the 'dim' column).",
        "  - {OMIC} = data type abbreviation (legend below).",
        "  - {hash} = a short hash of the feature name (this is what hides the name publicly).",
        "OMIC legend:",
        "  SIG  = immune / transcriptomic signatures      RPPA = protein (proteomic)",
        "  EXPR = gene expression                          IMM  = immune cell fractions",
        "  CNV  = copy-number variation                    MUT  = mutations",
        "",
        "COLUMN DICTIONARY (Decoder sheet)",
        "  website_code   The hashed code exactly as shown on the website.",
        "  dim            UNI2 dimension number (0-1535).",
        "  omic           Data type of the molecular feature (see legend).",
        "  feature_name   The REAL molecular feature this dimension correlates with (the decode).",
        "  program        Pathway / Hallmark program label for the dimension ('none' if unlabeled).",
        "  rho            Within-cancer Spearman correlation (dimension vs feature). Sign matters:",
        "                 + = higher dimension goes with higher feature; - = inverse.",
        "  q_value        Benjamini-Hochberg FDR-adjusted significance of that correlation.",
        "  n              Number of patients the correlation is computed over.",
        "  n_models       Robustness: in how many of 6 foundation models this link is top-20% (/6).",
        "  surv_hr        Hazard ratio for overall survival (stratified Cox); >1 = higher risk.",
        "  surv_p         P-value for that survival association.",
        "  surv_sig       TRUE if the survival association is significant.",
        "  on_tiles_page  TRUE if this dimension has tile montages on the website Tiles page.",
        "  montage_base   Montage file (figures/galleries/) -- top cases by raw activation.",
        "  montage_beyond Montage file -- ranked WITHIN cancer, excluding colon/rectum/liver.",
        "",
        "THE TILE MONTAGES",
        "Each montage is one dimension: 10 patient cases x the 100 image tiles that most activate",
        "that dimension, laid out as ten 10x10 blocks (one block per case).",
        "  - 'base'   ranks cases by raw activation and is dominated by liver and colon tissue.",
        "  - 'beyond' ranks cases WITHIN each cancer and drops colon, rectum and liver, so it shows",
        "             the same dimension in the other cancers. Tissue mix is roughly:",
        "             stomach 37%, cervix 26%, adrenocortical 20%, esophageal 13%, bile-duct 4%.",
        "How to view: open the live Tiles page (codes match this sheet), or open the JPEG files named",
        "in montage_base / montage_beyond under figures/galleries/.",
        "",
        "CAVEATS",
        "  - These are correlations, not causation.",
        "  - Correlations are within-cancer (cancer type regressed out), so they are not driven by",
        "    simply telling cancers apart.",
        "  - 'Most-activating tiles' are illustrative of what a dimension keys on, not a label.",
        "  - A few 'beyond' montages have fewer than 10 case blocks (limited slide availability).",
    ]


def main():
    if not os.path.exists(CATALOG):
        raise SystemExit(f"missing {CATALOG}")
    dec = build_decoder()
    guide = pd.DataFrame({"How to read this workbook": guide_lines()})

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        dec.to_excel(xl, sheet_name="Decoder", index=False)
        guide.to_excel(xl, sheet_name="How to read", index=False)

        # readability: freeze header, widen columns
        wsd = xl.sheets["Decoder"]
        wsd.freeze_panes = "A2"
        widths = {"website_code": 22, "dim": 8, "omic": 18, "feature_name": 32,
                  "program": 26, "rho": 8, "q_value": 12, "n": 7, "n_models": 10,
                  "surv_hr": 9, "surv_p": 11, "surv_sig": 9, "on_tiles_page": 13,
                  "montage_base": 18, "montage_beyond": 20}
        for i, c in enumerate(COLS):
            wsd.column_dimensions[chr(65 + i)].width = widths.get(c, 14)
        xl.sheets["How to read"].column_dimensions["A"].width = 100

    n_mont = int(dec["on_tiles_page"].sum())
    print(f"wrote {OUT}: {len(dec)} decoder rows ({n_mont} with montages)", flush=True)


if __name__ == "__main__":
    main()
