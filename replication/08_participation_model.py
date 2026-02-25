"""
============================================================================
08_participation_model.py
============================================================================
Selective Participation and Baseline Disadvantage (Section 6.4)

This script estimates the relationship between baseline county
characteristics and LCP participation intensity, supporting the paper's
claim that "counties with higher initial income ratios accumulated larger
transaction volumes."

Two-part model:
  Part 1: Probit/logit — does a county transact in a given year?
  Part 2: OLS — conditional on transacting, how much area is traded?

Input:  data/county_panel.csv
Output: output/tables/table_participation.tex (console summary)
============================================================================
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")
from plot_style import stars, fmt_coef_tex, fmt_se_tex

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "county_panel.csv")
TAB_DIR = os.path.join(BASE_DIR, "output", "tables")
os.makedirs(TAB_DIR, exist_ok=True)

# ============================================================================
# Load data
# ============================================================================
df = pd.read_csv(DATA_FILE)
print(f"Loaded: {df.shape[0]} obs, {df['county_id'].nunique()} counties")

# ============================================================================
# 1. BASELINE CHARACTERISTICS AND ADOPTION TIMING
# ============================================================================
print("\n" + "=" * 70)
print("1. BASELINE CHARACTERISTICS AND ADOPTION TIMING")
print("=" * 70)

baseline = df[df['year'] == 2009].copy()
baseline['ever_treated'] = (baseline['first_treat_year'] > 0).astype(int)
baseline['adoption_year'] = baseline['first_treat_year'].replace(0, np.nan)

# Correlation: baseline income ratio vs. adoption year (among treated)
treated_bl = baseline[baseline['ever_treated'] == 1].dropna(subset=['adoption_year'])
r, p = stats.pearsonr(treated_bl['urban_rural_income_ratio'], treated_bl['adoption_year'])
print(f"Corr(baseline ratio, adoption year): r={r:.3f}, p={p:.3f}")
# Negative r means higher ratio → earlier adoption

# ============================================================================
# 2. BASELINE RATIO AND CUMULATIVE TRADING INTENSITY
# ============================================================================
print("\n" + "=" * 70)
print("2. BASELINE RATIO AND CUMULATIVE TRADING INTENSITY")
print("=" * 70)

# Merge baseline characteristics with end-of-sample cumulative trading
final_year = df[df['year'] == 2020][['county_id', 'ln_cum_area']].copy()
final_year.rename(columns={'ln_cum_area': 'final_ln_cum_area'}, inplace=True)
bl_merged = baseline.merge(final_year, on='county_id', how='left')

# OLS: final cumulative area ~ baseline ratio + controls
X_vars = ['urban_rural_income_ratio', 'gdp_pc', 'urbanization_rate']
bl_reg = bl_merged.dropna(subset=X_vars + ['final_ln_cum_area'])

# Simple OLS with statsmodels-like output via pyfixest
# Create a temp dataset for cross-section regression
bl_reg['const'] = 1
for v in X_vars:
    print(f"  {v}: mean={bl_reg[v].mean():.2f}, std={bl_reg[v].std():.2f}")

r2, p2 = stats.pearsonr(bl_reg['urban_rural_income_ratio'], bl_reg['final_ln_cum_area'])
print(f"\nCorr(baseline ratio, final ln_cum_area): r={r2:.3f}, p={p2:.3f}")

# ============================================================================
# 3. PANEL PARTICIPATION MODEL
# ============================================================================
print("\n" + "=" * 70)
print("3. PANEL PARTICIPATION MODEL")
print("=" * 70)

# For each county-year, define participation as having positive annual area
df['participates'] = (df['ticket_area'] > 0).astype(int)

# Merge baseline ratio
baseline_ratio = df[df['year'] == 2009][['county_id', 'urban_rural_income_ratio']].copy()
baseline_ratio.rename(columns={'urban_rural_income_ratio': 'baseline_ratio'}, inplace=True)
df_part = df.merge(baseline_ratio, on='county_id', how='left')

# Only look at treated counties in post-treatment years
df_post = df_part[(df_part['first_treat_year'] > 0) & (df_part['year'] >= df_part['first_treat_year'])].copy()

print(f"Post-treatment obs: {len(df_post)}")
print(f"Participation rate: {df_post['participates'].mean():.3f}")

# Part 1: Linear probability model — participation ~ baseline_ratio + controls + year FE
m1 = pf.feols(
    "participates ~ baseline_ratio + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | year",
    data=df_post, vcov={"CRV1": "county_id"},
)
print("\nPart 1: Participation (LPM)")
print(f"  baseline_ratio: coef={m1.coef()['baseline_ratio']:.4f}, "
      f"se={m1.se()['baseline_ratio']:.4f}, p={m1.pvalue()['baseline_ratio']:.4f}")
for v in ['ln_gdp_pc', 'ln_fiscal_rev', 'ln_urban_rate']:
    print(f"  {v}: coef={m1.coef()[v]:.4f}, p={m1.pvalue()[v]:.4f}")

# Part 2: Conditional on participation, log area ~ baseline_ratio + controls + year FE
df_active = df_post[df_post['participates'] == 1].copy()
m2 = pf.feols(
    "ln_cum_area ~ baseline_ratio + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | year",
    data=df_active, vcov={"CRV1": "county_id"},
)
print("\nPart 2: Cumulative area (conditional on participation)")
print(f"  baseline_ratio: coef={m2.coef()['baseline_ratio']:.4f}, "
      f"se={m2.se()['baseline_ratio']:.4f}, p={m2.pvalue()['baseline_ratio']:.4f}")
for v in ['ln_gdp_pc', 'ln_fiscal_rev', 'ln_urban_rate']:
    print(f"  {v}: coef={m2.coef()[v]:.4f}, p={m2.pvalue()[v]:.4f}")

# ============================================================================
# 4. GENERATE LATEX TABLE
# ============================================================================
print("\n" + "=" * 70)
print("4. PARTICIPATION TABLE (LaTeX)")
print("=" * 70)

vars_list = ['baseline_ratio', 'ln_fiscal_rev', 'ln_urban_rate']
var_labels = {
    'baseline_ratio': 'Baseline income ratio',
    'ln_fiscal_rev': 'Log fiscal revenue',
    'ln_urban_rate': 'Log urbanization rate',
}

tex = []
tex.append(r"\begin{table}[ht!]")
tex.append(r"\centering")
tex.append(r"\caption{Selective Participation: Baseline Inequality and Trading Intensity}")
tex.append(r"\label{tab:participation}")
tex.append(r"\begin{threeparttable}")
tex.append(r"\begin{tabular}{lcc}")
tex.append(r"\toprule")
tex.append(r"& Participation (LPM) & ln(Cumulative area) \\")
tex.append(r"& (1) & (2) \\")
tex.append(r"\midrule")

for v in vars_list:
    label = var_labels.get(v, v)
    c1 = fmt_coef_tex(m1.coef()[v], m1.pvalue()[v])
    s1 = fmt_se_tex(m1.se()[v])
    c2 = fmt_coef_tex(m2.coef()[v], m2.pvalue()[v])
    s2 = fmt_se_tex(m2.se()[v])
    tex.append(f"{label} & {c1} & {c2} \\\\")
    tex.append(f" & {s1} & {s2} \\\\[3pt]")

tex.append(r"\midrule")
tex.append(r"Year FE & Yes & Yes \\")
tex.append(f"Observations & {len(df_post)} & {len(df_active)} \\\\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\begin{tablenotes}[flushleft]")
tex.append(r"\item \small Column~(1): linear probability model for annual participation (positive transaction area). Column~(2): OLS for log cumulative transaction area, conditional on participation. Sample restricted to treated counties in post-treatment years. Standard errors clustered by county.")
tex.append(r"\item $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$")
tex.append(r"\end{tablenotes}")
tex.append(r"\end{threeparttable}")
tex.append(r"\end{table}")

part_path = os.path.join(TAB_DIR, "table_participation.tex")
with open(part_path, "w") as f:
    f.write("\n".join(tex))
print(f"Participation table saved to {part_path}")

print("\n" + "=" * 70)
print("PARTICIPATION MODEL ANALYSIS COMPLETE")
print("=" * 70)
