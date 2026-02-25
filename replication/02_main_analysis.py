"""
============================================================================
02_main_analysis.py
============================================================================
Main Econometric Analysis: Event Study and TWFE Estimates

This script replicates the core results of the paper:
  1. Event-study difference-in-differences (Figure 2 / Table 4 cols 1-2)
  2. TWFE with treatment intensity: annual and cumulative (Table 4 cols 3-6)
  3. Publication-ready event study figure

Input:  data/county_panel.csv (from 01_data_preparation.py)
Output: output/figures/event_study_main.pdf
        output/figures/event_study_main.png
        output/tables/table_main_did.tex
        output/event_study_results.csv
        output/event_study_results_controls.csv
============================================================================
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
from plot_style import *

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "county_panel.csv")
FIG_DIR = os.path.join(BASE_DIR, "output", "figures")
TAB_DIR = os.path.join(BASE_DIR, "output", "tables")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ============================================================================
# Load data
# ============================================================================
df = pd.read_csv(DATA_FILE)
print(f"Loaded panel: {df.shape[0]} obs, {df['county_id'].nunique()} counties, "
      f"{df['year'].nunique()} years ({df['year'].min()}-{df['year'].max()})")


# ============================================================================
# PART A: EVENT-STUDY ESTIMATION
# ============================================================================
print("\n" + "=" * 70)
print("PART A: EVENT-STUDY ESTIMATION")
print("=" * 70)

# Prepare event-study data:
#   - Clip relative time to [-4, 10] (bin endpoints)
#   - Never-treated counties get rel_time = -1 (always in reference group)
df_es = df.copy()
df_es["rt"] = df_es["rel_time"].copy()
df_es.loc[df_es["rt"] < -4, "rt"] = -4
df_es.loc[df_es["rt"] > 10, "rt"] = 10
df_es.loc[df_es["first_treat_year"] == 0, "rt"] = -1.0
df_es["rt"] = df_es["rt"].astype(float)

# --- Model 1: Event study without controls ---
es_base = pf.feols(
    "urban_rural_income_ratio ~ i(rt, ref=-1.0) | county_id + year",
    data=df_es,
    vcov={"CRV1": "county_id"},
)

# --- Model 2: Event study with controls ---
es_ctrl = pf.feols(
    "urban_rural_income_ratio ~ i(rt, ref=-1.0) + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | county_id + year",
    data=df_es,
    vcov={"CRV1": "county_id"},
)


def extract_event_study(model, data, ref_period=-1):
    """Extract event-study coefficients from a pyfixest model.

    Uses robust extract_es_coefs from plot_style module.
    Returns DataFrame with columns: rel_time, coef, se, ci_lower, ci_upper
    for compatibility with downstream code.
    """
    es_coefs = extract_es_coefs(model, data)
    results = []
    for rt, coef, se in es_coefs:
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        results.append({
            "rel_time": rt,
            "coef": coef,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })
    return pd.DataFrame(results).sort_values("rel_time").reset_index(drop=True)


res_base = extract_event_study(es_base, df_es)
res_ctrl = extract_event_study(es_ctrl, df_es)

print("\nEvent-Study Coefficients (baseline, no controls):")
print(res_base.to_string(index=False))

# Key results
post = res_base[res_base["rel_time"] >= 0]
print(f"\n--- Key Results ---")
print(f"Effect at adoption (e=0):   {res_base.loc[res_base.rel_time==0, 'coef'].values[0]:.4f}")
print(f"Effect at 5 years (e=5):    {res_base.loc[res_base.rel_time==5, 'coef'].values[0]:.4f}")
print(f"Effect at 10 years (e=10):  {res_base.loc[res_base.rel_time==10, 'coef'].values[0]:.4f}")
print(f"Average post-treatment ATT: {post['coef'].mean():.4f}")

# Save event-study results
res_base.to_csv(os.path.join(OUT_DIR, "event_study_results.csv"), index=False)
res_ctrl.to_csv(os.path.join(OUT_DIR, "event_study_results_controls.csv"), index=False)


# ============================================================================
# PART B: EVENT-STUDY FIGURE (Publication-Ready)
# ============================================================================
print("\n" + "=" * 70)
print("PART B: EVENT-STUDY FIGURE")
print("=" * 70)

apply_style()

fig, ax = plt.subplots(figsize=(6.5, 4))

# Plot baseline estimates with 95% CI band
plot_es(
    ax,
    res_base["rel_time"].values,
    res_base["coef"].values,
    res_base["se"].values,
    color=C_BLUE,
    marker='o',
    label='Without controls',
    band=True,
)

# Plot with-controls estimates (offset slightly for visibility)
plot_es(
    ax,
    res_ctrl["rel_time"].values + 0.15,
    res_ctrl["coef"].values,
    res_ctrl["se"].values,
    color=C_ORANGE,
    marker='D',
    label='With controls',
    band=False,
)

# Apply consistent styling
style_axis(ax, ylabel='Effect on urban\u2013rural income ratio')
ax.legend(frameon=False, loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "event_study_main.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "event_study_main.pdf"), dpi=300, bbox_inches="tight")
plt.close()
print("Event study figure saved.")


# ============================================================================
# PART C: TWFE ESTIMATES — BINARY AND TREATMENT INTENSITY (Table 4)
# ============================================================================
print("\n" + "=" * 70)
print("PART C: TWFE ESTIMATES (Table 4)")
print("=" * 70)

# Column 1: Binary treatment, no controls
m1 = pf.feols(
    "urban_rural_income_ratio ~ treated | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
# Column 2: Binary treatment, with controls
m2 = pf.feols(
    "urban_rural_income_ratio ~ treated + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
# Column 3: Annual intensity, no controls
m3 = pf.feols(
    "urban_rural_income_ratio ~ ln_annual_area | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
# Column 4: Annual intensity, with controls
m4 = pf.feols(
    "urban_rural_income_ratio ~ ln_annual_area + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
# Column 5: Cumulative intensity, no controls
m5 = pf.feols(
    "urban_rural_income_ratio ~ ln_cum_area | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
# Column 6: Cumulative intensity, with controls
m6 = pf.feols(
    "urban_rural_income_ratio ~ ln_cum_area + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)

models = [
    ("(1) Binary", "treated", m1),
    ("(2) Binary+Ctrl", "treated", m2),
    ("(3) Annual", "ln_annual_area", m3),
    ("(4) Annual+Ctrl", "ln_annual_area", m4),
    ("(5) Cumulative", "ln_cum_area", m5),
    ("(6) Cumul+Ctrl", "ln_cum_area", m6),
]

print(f"\n{'Specification':<25} {'Coef':>10} {'SE':>10} {'p-value':>10}")
print("-" * 60)
for label, var, model in models:
    c = model.coef()[var]
    s = model.se()[var]
    p = model.pvalue()[var]
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"{label:<25} {c:>10.4f} {s:>10.4f} {p:>10.4f} {sig}")


# ============================================================================
# PART D: GENERATE LaTeX TABLE
# ============================================================================
print("\n" + "=" * 70)
print("PART D: GENERATING LaTeX TABLE")
print("=" * 70)

tex = []
tex.append(r"\begin{table}[ht!]")
tex.append(r"\centering")
tex.append(r"\caption{Staggered Difference-in-Differences Estimates: Binary Treatment and Treatment Intensity}")
tex.append(r"\label{tab:main_did}")
tex.append(r"\begin{threeparttable}")
tex.append(r"\begin{tabular}{l*{6}{c}}")
tex.append(r"\toprule")
tex.append(r" & \multicolumn{2}{c}{Binary Treatment} & \multicolumn{2}{c}{Annual Intensity} & \multicolumn{2}{c}{Cumulative Intensity} \\")
tex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
tex.append(r" & (1) & (2) & (3) & (4) & (5) & (6) \\")
tex.append(r"\midrule")

# Binary treatment row
c1, s1, p1 = m1.coef()["treated"], m1.se()["treated"], m1.pvalue()["treated"]
c2, s2, p2 = m2.coef()["treated"], m2.se()["treated"], m2.pvalue()["treated"]
tex.append(f"Treated ($D_{{it}}$) & ${c1:.3f}${stars(p1)} & ${c2:.3f}${stars(p2)} & & & & \\\\")
tex.append(f"  & ({s1:.3f}) & ({s2:.3f}) & & & & \\\\")

# Annual intensity row
c3, s3, p3 = m3.coef()["ln_annual_area"], m3.se()["ln_annual_area"], m3.pvalue()["ln_annual_area"]
c4, s4, p4 = m4.coef()["ln_annual_area"], m4.se()["ln_annual_area"], m4.pvalue()["ln_annual_area"]
tex.append(f"ln(Annual area + 1) & &  & ${c3:.3f}${stars(p3)} & ${c4:.3f}${stars(p4)} & & \\\\")
tex.append(f"  & &  & ({s3:.3f}) & ({s4:.3f}) & & \\\\")

# Cumulative intensity row
c5, s5, p5 = m5.coef()["ln_cum_area"], m5.se()["ln_cum_area"], m5.pvalue()["ln_cum_area"]
c6, s6, p6 = m6.coef()["ln_cum_area"], m6.se()["ln_cum_area"], m6.pvalue()["ln_cum_area"]
tex.append(f"ln(Cumulative area + 1) & & & &  & ${c5:.3f}${stars(p5)} & ${c6:.3f}${stars(p6)} \\\\")
tex.append(f"  & & & &  & ({s5:.3f}) & ({s6:.3f}) \\\\")

tex.append(r"\midrule")
tex.append(r"Controls & No & Yes & No & Yes & No & Yes \\")
tex.append(r"County FE & Yes & Yes & Yes & Yes & Yes & Yes \\")
tex.append(r"Year FE & Yes & Yes & Yes & Yes & Yes & Yes \\")

# Get observation counts (should be same for all models)
n_obs = len(df)
n_clust = df["county_id"].nunique()

tex.append(f"Observations & {n_obs} & {n_obs} & {n_obs} & {n_obs} & {n_obs} & {n_obs} \\\\")
tex.append(f"Clusters & {n_clust} & {n_clust} & {n_clust} & {n_clust} & {n_clust} & {n_clust} \\\\")

# R² (within) for each model
r2_1 = m1._r2_within
r2_2 = m2._r2_within
r2_3 = m3._r2_within
r2_4 = m4._r2_within
r2_5 = m5._r2_within
r2_6 = m6._r2_within
tex.append(f"$R^2$ (within) & {r2_1:.3f} & {r2_2:.3f} & {r2_3:.3f} & {r2_4:.3f} & {r2_5:.3f} & {r2_6:.3f} \\\\")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\begin{tablenotes}[flushleft]")
tex.append(r"\small")
tex.append(r"\item \textit{Notes:} Dependent variable is the urban--rural income ratio. Controls include log GDP per capita, log fiscal revenue, and log urbanization rate. Standard errors clustered at the county level in parentheses. $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$.")
tex.append(r"\end{tablenotes}")
tex.append(r"\end{threeparttable}")
tex.append(r"\end{table}")

table_path = os.path.join(TAB_DIR, "table_main_did.tex")
with open(table_path, "w") as f:
    f.write("\n".join(tex))
print(f"LaTeX table saved to: {table_path}")

print("\n" + "=" * 70)
print("MAIN ANALYSIS COMPLETE")
print("=" * 70)
