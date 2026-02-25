"""
============================================================================
03_robustness_and_mechanisms.py
============================================================================
Robustness Checks and Mechanism Analysis

This script replicates:
  1. Goodman-Bacon decomposition (Figure 4)
  2. Cohort-specific ATT estimates (Table 5)
  3. Alternative control groups and specifications (Table 6)
  4. Income decomposition event studies — rural vs urban (Figure 5)
  5. Selective participation (balance) test (Table 7)

Input:  data/county_panel.csv
        output/event_study_results.csv (from 02_main_analysis.py)
Output: output/figures/bacon_decomposition.png
        output/figures/income_decomposition_event_study.png
        output/figures/income_decomposition_event_study.pdf
        output/tables/table_cohort_att.tex
        output/tables/table_robustness.tex
        output/tables/table_mechanism_income.tex
        output/tables/table_balance.tex
============================================================================
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
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
# Load data and apply style
# ============================================================================
df = pd.read_csv(DATA_FILE)
print(f"Loaded: {df.shape[0]} obs, {df['county_id'].nunique()} counties")

apply_style()


# ############################################################################
# SECTION 1: GOODMAN-BACON DECOMPOSITION
# ############################################################################
print("\n" + "=" * 70)
print("1. GOODMAN-BACON DECOMPOSITION (APPROXIMATE)")
print("=" * 70)

cohort_groups = sorted(df[df["first_treat_year"] > 0]["first_treat_year"].unique())
never_treated = df[df["first_treat_year"] == 0].copy()

bacon_results = []

# Type A: Treated cohort vs. never-treated (clean 2x2 DD)
for g in cohort_groups:
    treated_g = df[df["first_treat_year"] == g].copy()
    combined = pd.concat([treated_g, never_treated])
    combined["post"] = np.where(
        (combined["first_treat_year"] == g) & (combined["year"] >= g), 1, 0
    )
    try:
        m = pf.feols(
            "urban_rural_income_ratio ~ post | county_id + year",
            data=combined, vcov={"CRV1": "county_id"},
        )
        n_t = treated_g["county_id"].nunique()
        n_c = never_treated["county_id"].nunique()
        w = n_t * n_c / ((n_t + n_c) ** 2)
        bacon_results.append({
            "comparison": f"Cohort {g} vs Never",
            "type": "Treated vs Never",
            "estimate": m.coef()["post"],
            "se": m.se()["post"],
            "weight_raw": w,
        })
    except Exception:
        pass

# Type B: Early vs. late treated (potentially problematic)
for i, g_early in enumerate(cohort_groups):
    for g_late in cohort_groups[i + 1:]:
        early = df[df["first_treat_year"] == g_early].copy()
        late = df[df["first_treat_year"] == g_late].copy()
        combined = pd.concat([early, late])
        combined["post"] = np.where(
            (combined["first_treat_year"] == g_early) & (combined["year"] >= g_early), 1, 0
        )
        try:
            m = pf.feols(
                "urban_rural_income_ratio ~ post | county_id + year",
                data=combined, vcov={"CRV1": "county_id"},
            )
            n_e = early["county_id"].nunique()
            n_l = late["county_id"].nunique()
            w = n_e * n_l / ((n_e + n_l) ** 2)
            bacon_results.append({
                "comparison": f"Cohort {g_early} vs {g_late}",
                "type": "Early vs Late",
                "estimate": m.coef()["post"],
                "se": m.se()["post"],
                "weight_raw": w,
            })
        except Exception:
            pass

bacon_df = pd.DataFrame(bacon_results)
bacon_df["weight"] = bacon_df["weight_raw"] / bacon_df["weight_raw"].sum()
bacon_df["weighted_est"] = bacon_df["estimate"] * bacon_df["weight"]

print("\nDecomposition components:")
for _, row in bacon_df.iterrows():
    print(f"  {row['comparison']}: est={row['estimate']:.4f}, weight={row['weight']:.3f}")

by_type = bacon_df.groupby("type").agg(
    total_weight=("weight", "sum"),
    weighted_sum=("weighted_est", "sum"),
)
by_type["avg_estimate"] = by_type["weighted_sum"] / by_type["total_weight"]
print(f"\nBy type:\n{by_type.to_string()}")
print(f"\nWeighted sum (≈ TWFE): {bacon_df['weighted_est'].sum():.4f}")

# Bacon decomposition plot
fig, ax = plt.subplots(figsize=(6.5, 4))

# Plot by type with unified colors
for ctype in bacon_df["type"].unique():
    subset = bacon_df[bacon_df["type"] == ctype]
    if ctype == "Treated vs Never":
        marker, color = 'o', C_BLUE
    else:
        marker, color = 's', C_ORANGE
    ax.scatter(
        subset["estimate"], subset["weight"],
        s=np.maximum(subset["weight"] * 800, 20),
        color=color, marker=marker, alpha=0.7,
        edgecolors="white", linewidth=0.5, label=ctype, zorder=3
    )

# Add red dashed line for overall TWFE
overall_twfe = bacon_df["weighted_est"].sum()
ax.axvline(x=overall_twfe, color='red', linewidth=1.2, linestyle='--', alpha=0.7, zorder=2)

# Apply style then remove the -0.5 vline that style_axis adds
style_axis(ax, ylabel='Weight', xlabel='2×2 DiD estimate')
for line in ax.get_lines():
    if hasattr(line, 'get_xdata'):
        xdata = line.get_xdata()
        if len(xdata) > 0 and abs(xdata[0] - (-0.5)) < 0.01:
            line.remove()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Custom legend with Line2D handles
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BLUE, markersize=7,
           label='Treated vs Never', markeredgecolor='white', markeredgewidth=0.5),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_ORANGE, markersize=7,
           label='Early vs Late', markeredgecolor='white', markeredgewidth=0.5),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "bacon_decomposition.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Bacon decomposition figure saved.")


# ############################################################################
# SECTION 2: COHORT-SPECIFIC ATT ESTIMATES
# ############################################################################
print("\n" + "=" * 70)
print("2. COHORT-SPECIFIC ATT ESTIMATES (Table 5)")
print("=" * 70)

cohort_results = []
for g in cohort_groups:
    g_data = df[(df["first_treat_year"] == g) | (df["first_treat_year"] == 0)].copy()
    g_data["post_g"] = np.where(
        (g_data["first_treat_year"] == g) & (g_data["year"] >= g), 1, 0
    )
    n = g_data[g_data["first_treat_year"] == g]["county_id"].nunique()
    try:
        m = pf.feols(
            "urban_rural_income_ratio ~ post_g | county_id + year",
            data=g_data, vcov={"CRV1": "county_id"},
        )
        cohort_results.append({
            "cohort": g, "n_counties": n,
            "att": m.coef()["post_g"], "se": m.se()["post_g"],
            "pval": m.pvalue()["post_g"],
        })
        s = "***" if m.pvalue()["post_g"] < 0.01 else "**" if m.pvalue()["post_g"] < 0.05 else "*" if m.pvalue()["post_g"] < 0.1 else ""
        print(f"  Cohort {g} (n={n}): ATT = {m.coef()['post_g']:.4f} (SE={m.se()['post_g']:.4f}) {s}")
    except Exception as e:
        print(f"  Cohort {g}: failed ({e})")

cohort_df = pd.DataFrame(cohort_results)
cohort_df.to_csv(os.path.join(OUT_DIR, "cohort_specific_att.csv"), index=False)

# Generate LaTeX table (Table 5) with booktabs
tab5 = []
tab5.append(r"\begin{threeparttable}")
tab5.append(r"\centering")
tab5.append(r"\caption{Cohort-Specific ATT Estimates}")
tab5.append(r"\label{tab:cohort_att}")
tab5.append(r"\begin{tabular}{lccc}")
tab5.append(r"\toprule")
tab5.append(r"Cohort & ATT & SE & $N$ (counties) \\")
tab5.append(r"\midrule")

for _, row in cohort_df.iterrows():
    att_fmt = fmt_coef_tex(row['att'], row['pval'])
    se_fmt = fmt_se_tex(row['se'])
    tab5.append(f"{int(row['cohort'])} & {att_fmt} & {se_fmt} & {int(row['n_counties'])} \\\\")

tab5.append(r"\bottomrule")
tab5.append(r"\end{tabular}")
tab5.append(r"\begin{tablenotes}[flushleft]")
tab5.append(r"\item \small Dependent variable: Urban--rural income ratio. Specification: binary treatment indicator with county and year fixed effects. Standard errors clustered by county.")
tab5.append(r"\item $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$")
tab5.append(r"\end{tablenotes}")
tab5.append(r"\end{threeparttable}")

tab5_path = os.path.join(TAB_DIR, "table_cohort_att.tex")
with open(tab5_path, "w") as f:
    f.write("\n".join(tab5))
print(f"\nTable 5 (Cohort ATT) saved to {tab5_path}")


# ############################################################################
# SECTION 3: ROBUSTNESS OF STAGGERED DiD ESTIMATES (Table 6)
# ############################################################################
print("\n" + "=" * 70)
print("3. ROBUSTNESS OF STAGGERED DiD ESTIMATES (Table 6)")
print("=" * 70)

robustness = []
robustness_n = []

# (a) Exclude 2009 cohort — counties treated from the first year of data,
#     lacking pre-treatment observations for the parallel-trends check.
df_ex09 = df[df["first_treat_year"] != 2009].copy()
m = pf.feols("urban_rural_income_ratio ~ treated | county_id + year",
             data=df_ex09, vcov={"CRV1": "county_id"})
robustness.append(("Excl. 2009 cohort", m.coef()["treated"], m.se()["treated"], m.pvalue()["treated"]))
robustness_n.append(len(df_ex09))

# (b) 2011+ cohorts only — drop both 2009 and 2010 early adopters,
#     retaining only cohorts with at least one full pre-treatment year.
df_2011 = df[(df["first_treat_year"] >= 2011) | (df["first_treat_year"] == 0)].copy()
m = pf.feols("urban_rural_income_ratio ~ treated | county_id + year",
             data=df_2011, vcov={"CRV1": "county_id"})
robustness.append(("2011+ cohorts", m.coef()["treated"], m.se()["treated"], m.pvalue()["treated"]))
robustness_n.append(len(df_2011))

# (c) Exclude 2011 cohort — the largest single treatment group (8 counties),
#     verifying that no single cohort drives the result.
df_ex11 = df[df["first_treat_year"] != 2011].copy()
m = pf.feols("urban_rural_income_ratio ~ treated | county_id + year",
             data=df_ex11, vcov={"CRV1": "county_id"})
robustness.append(("Excl. 2011 cohort", m.coef()["treated"], m.se()["treated"], m.pvalue()["treated"]))
robustness_n.append(len(df_ex11))

# (d) Extended controls — adds log population and primary-sector GDP share
#     beyond the baseline controls (log GDP pc, log fiscal revenue, log urbanization).
df["ln_pop"] = np.log(df["total_pop_registered(10k)"])
df["primary_share"] = df["GDP_primary_sector(10k)"] / df["GDP_total(10k)"]
m = pf.feols(
    "urban_rural_income_ratio ~ treated + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate"
    " + ln_pop + primary_share | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
robustness.append(("Extended controls", m.coef()["treated"], m.se()["treated"], m.pvalue()["treated"]))
robustness_n.append(len(df))

# (e) Log specification — dependent variable is ln(urban_rural_income_ratio),
#     with baseline controls.
df["lnratio"] = np.log(df["urban_rural_income_ratio"])
m = pf.feols(
    "lnratio ~ treated + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
robustness.append(("Log spec.", m.coef()["treated"], m.se()["treated"], m.pvalue()["treated"]))
robustness_n.append(len(df))

# ---- Print results ----
print(f"\n{'Specification':<25} {'ATT':>10} {'SE':>10} {'p':>10} {'N':>8}")
print("-" * 68)
for (name, est, se, p), n in zip(robustness, robustness_n):
    s = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"{name:<25} {est:>10.4f} {se:>10.4f} {p:>10.4f} {s:<3s} {n:>6}")

# ---- Generate LaTeX table (Table 6) with booktabs ----
tab6 = []
tab6.append(r"\begin{threeparttable}")
tab6.append(r"\centering")
tab6.append(r"\caption{Robustness of Staggered DiD Estimates}")
tab6.append(r"\label{tab:robustness}")
tab6.append(r"\begin{tabular}{lccccc}")
tab6.append(r"\toprule")
tab6.append(r"& \multicolumn{4}{c}{\textit{Urban--Rural Income Ratio}} & \textit{ln(Ratio)} \\")
tab6.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}")
tab6.append(r"& Excl.\ 2009 & 2011+ & Excl.\ 2011 & Extended & Log \\")
tab6.append(r"& cohort & cohorts & cohort & controls & spec. \\")
tab6.append(r"& (1) & (2) & (3) & (4) & (5) \\")
tab6.append(r"\midrule")

coefs_line = "Treated"
se_line = ""
for name, est, se, p in robustness:
    coefs_line += f" & {fmt_coef_tex(est, p)}"
    se_line += f" & {fmt_se_tex(se)}"
tab6.append(coefs_line + r" \\")
tab6.append(se_line + r" \\")

tab6.append(r"\midrule")
tab6.append(r"Controls & No & No & No & Yes & Yes \\")
tab6.append(r"County FE & Yes & Yes & Yes & Yes & Yes \\")
tab6.append(r"Year FE & Yes & Yes & Yes & Yes & Yes \\")

obs_line = "Observations"
for n in robustness_n:
    obs_line += f" & {n}"
tab6.append(obs_line + r" \\")

tab6.append(r"\bottomrule")
tab6.append(r"\end{tabular}")
tab6.append(r"\begin{tablenotes}[flushleft]")
tab6.append(r"\item \small SEs clustered by county. Cols 1--3: binary treatment, no controls. Col 4: adds log population and primary-sector GDP share. Col 5: log of income ratio with baseline controls.")
tab6.append(r"\item $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$")
tab6.append(r"\end{tablenotes}")
tab6.append(r"\end{threeparttable}")

tab6_path = os.path.join(TAB_DIR, "table_robustness.tex")
with open(tab6_path, "w") as f:
    f.write("\n".join(tab6))
print(f"\nTable 6 saved to {tab6_path}")


# ############################################################################
# SECTION 3B: LEAVE-ONE-OUT SENSITIVITY (Appendix Table A.4)
# ############################################################################
print("\n" + "=" * 70)
print("3B. LEAVE-ONE-OUT SENSITIVITY (Appendix Table A.4)")
print("=" * 70)

# Drop each never-treated county in turn to check sensitivity
never_treated_ids = sorted(df[df["first_treat_year"] == 0]["county_id"].unique())
loo_results = []

for drop_id in never_treated_ids:
    name = df[df["county_id"] == drop_id]["county_name"].iloc[0]
    df_loo = df[df["county_id"] != drop_id].copy()
    m = pf.feols(
        "urban_rural_income_ratio ~ treated | county_id + year",
        data=df_loo, vcov={"CRV1": "county_id"},
    )
    c, s, p = m.coef()["treated"], m.se()["treated"], m.pvalue()["treated"]
    star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    loo_results.append({
        "county": name, "id": drop_id, "coef": c, "se": s, "pval": p,
        "n_counties": df_loo["county_id"].nunique(), "n_obs": len(df_loo)
    })
    print(f"  Drop {name} ({drop_id}): coef={c:.4f}{star}, se={s:.4f}, p={p:.4f}")

# Generate LaTeX table with booktabs
loo_tex = []
loo_tex.append(r"\begin{threeparttable}")
loo_tex.append(r"\centering")
loo_tex.append(r"\caption{Leave-One-Out Sensitivity: Dropping Each Never-Treated County}")
loo_tex.append(r"\label{tab:loo}")
loo_tex.append(r"\begin{tabular}{lccccc}")
loo_tex.append(r"\toprule")
loo_tex.append(r"Dropped County & ATT & SE & $p$-value & Counties & Obs. \\")
loo_tex.append(r"\midrule")

# Pinyin names for LaTeX compatibility
pinyin_map = {
    500104: "Dadukou District", 500105: "Jiangbei District",
    500106: "Shapingba District", 500108: "Nan'an District",
    500111: "Dazu District", 500115: "Changshou District",
}
for r in loo_results:
    name_en = pinyin_map.get(r["id"], r["county"])
    star = "$^{***}$" if r["pval"] < 0.01 else "$^{**}$" if r["pval"] < 0.05 else "$^{*}$" if r["pval"] < 0.1 else ""
    loo_tex.append(
        f"{name_en} & ${r['coef']:.3f}${star} & $({r['se']:.3f})$ & "
        f"${r['pval']:.3f}$ & {r['n_counties']} & {r['n_obs']} \\\\"
    )

loo_tex.append(r"\midrule")
loo_tex.append(r"Full sample & $-0.062^{*}$ & $(0.031)$ & $0.050$ & 37 & 444 \\")
loo_tex.append(r"\bottomrule")
loo_tex.append(r"\end{tabular}")
loo_tex.append(r"\begin{tablenotes}[flushleft]")
loo_tex.append(r"\item \small Dependent variable: Urban--Rural Income Ratio. Each row drops one never-treated county.")
loo_tex.append(r"\item County and year fixed effects. Standard errors clustered by county.")
loo_tex.append(r"\item $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$")
loo_tex.append(r"\end{tablenotes}")
loo_tex.append(r"\end{threeparttable}")

loo_path = os.path.join(TAB_DIR, "table_loo.tex")
with open(loo_path, "w") as f:
    f.write("\n".join(loo_tex))
print(f"\nLeave-one-out table saved to {loo_path}")


# ############################################################################
# SECTION 3C: WILD CLUSTER BOOTSTRAP INFERENCE
# ############################################################################
print("\n" + "=" * 70)
print("3C. WILD CLUSTER BOOTSTRAP INFERENCE")
print("=" * 70)

# Baseline TWFE
m_base = pf.feols(
    "urban_rural_income_ratio ~ treated | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
wb_base = m_base.wildboottest(param="treated", cluster="county_id", reps=9999, seed=42)
print(f"Baseline WCB p-value: {wb_base['Pr(>|t|)']:.4f}")

# With controls
m_ctrl = pf.feols(
    "urban_rural_income_ratio ~ treated + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
wb_ctrl = m_ctrl.wildboottest(param="treated", cluster="county_id", reps=9999, seed=42)
print(f"With controls WCB p-value: {wb_ctrl['Pr(>|t|)']:.4f}")

# Cumulative intensity
m_cum = pf.feols(
    "urban_rural_income_ratio ~ ln_cum_area | county_id + year",
    data=df, vcov={"CRV1": "county_id"},
)
wb_cum = m_cum.wildboottest(param="ln_cum_area", cluster="county_id", reps=9999, seed=42)
print(f"Cumulative intensity WCB p-value: {wb_cum['Pr(>|t|)']:.4f}")


# ############################################################################
# SECTION 4: INCOME DECOMPOSITION EVENT STUDIES
# ############################################################################
print("\n" + "=" * 70)
print("4. INCOME DECOMPOSITION EVENT STUDIES (Figure 5)")
print("=" * 70)

df_es = df.copy()
df_es["rt"] = df_es["rel_time"].copy()
df_es.loc[df_es["rt"] < -4, "rt"] = -4
df_es.loc[df_es["rt"] > 10, "rt"] = 10
df_es.loc[df_es["first_treat_year"] == 0, "rt"] = -1.0


def run_event_study(data, outcome):
    """Run event study for a given outcome variable using extract_es_coefs."""
    m = pf.feols(
        f"{outcome} ~ i(rt, ref=-1.0) | county_id + year",
        data=data, vcov={"CRV1": "county_id"},
    )
    es_coefs = extract_es_coefs(m, data)
    rows = []
    for rt, coef, se in es_coefs:
        rows.append({
            "rel_time": rt, "coef": coef, "se": se,
            "ci_lo": coef - 1.96 * se, "ci_hi": coef + 1.96 * se,
        })
    return pd.DataFrame(rows).sort_values("rel_time")


rural_es = run_event_study(df_es, "ln_rural_income")
urban_es = run_event_study(df_es, "ln_urban_income")
main_es = pd.read_csv(os.path.join(OUT_DIR, "event_study_results.csv"))

print("Rural income event study (log):")
print(rural_es[["rel_time", "coef", "se"]].to_string(index=False))
print("\nUrban income event study (log):")
print(urban_es[["rel_time", "coef", "se"]].to_string(index=False))

# Three-panel figure with unified style
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel (a): Urban-rural income ratio
plot_es(axes[0], main_es["rel_time"], main_es["coef"], main_es["se"],
        color=C_BLUE, marker='o', label=None, band=True)
style_axis(axes[0], ylabel='Effect', xlabel='Years since first transaction')
axes[0].set_title('(a) Urban–rural income ratio', fontsize=10)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Panel (b): Log rural income
plot_es(axes[1], rural_es["rel_time"], rural_es["coef"], rural_es["se"],
        color=C_GREEN, marker='o', label=None, band=True)
style_axis(axes[1], ylabel='Effect', xlabel='Years since first transaction')
axes[1].set_title('(b) Log rural disposable income', fontsize=10)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Panel (c): Log urban income
plot_es(axes[2], urban_es["rel_time"], urban_es["coef"], urban_es["se"],
        color=C_ORANGE, marker='o', label=None, band=True)
style_axis(axes[2], ylabel='Effect', xlabel='Years since first transaction')
axes[2].set_title('(c) Log urban disposable income', fontsize=10)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "income_decomposition_event_study.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "income_decomposition_event_study.pdf"), dpi=300, bbox_inches="tight")
plt.close()
print("Income decomposition event study figure saved.")


# ############################################################################
# SECTION 5: SELECTIVE PARTICIPATION (BALANCE TABLE)
# ############################################################################
print("\n" + "=" * 70)
print("5. SELECTIVE PARTICIPATION — BALANCE TABLE (Table 7)")
print("=" * 70)

baseline = df[df["year"] == 2009].copy()
baseline["ever_treated"] = (baseline["first_treat_year"] > 0).astype(int)

balance_vars = [
    ("urban_rural_income_ratio", "Urban-rural income ratio"),
    ("gdp_pc", "GDP per capita"),
    ("fiscal_rev", "Fiscal revenue"),
    ("urbanization_rate", "Urbanization rate (%)"),
]

print(f"\n{'Variable':<30} {'Treated':>10} {'Never':>10} {'Diff':>10} {'p-value':>10}")
print("-" * 75)
for var, label in balance_vars:
    t_vals = baseline.loc[baseline["ever_treated"] == 1, var]
    n_vals = baseline.loc[baseline["ever_treated"] == 0, var]
    t_stat, p_val = stats.ttest_ind(t_vals, n_vals)
    diff = t_vals.mean() - n_vals.mean()
    s = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    print(f"{label:<30} {t_vals.mean():>10.2f} {n_vals.mean():>10.2f} {diff:>10.2f} {p_val:>10.3f} {s}")


# ############################################################################
# SECTION 6: BASELINE BALANCE TABLE (LaTeX)
# ############################################################################
print("\n" + "=" * 70)
print("6. BASELINE BALANCE TABLE (LaTeX — table_balance.tex)")
print("=" * 70)

bal_rows = []
for var, label in balance_vars:
    t_vals = baseline.loc[baseline["ever_treated"] == 1, var]
    n_vals = baseline.loc[baseline["ever_treated"] == 0, var]
    t_stat, p_val = stats.ttest_ind(t_vals, n_vals)
    diff = t_vals.mean() - n_vals.mean()
    bal_rows.append({
        "label": label, "treated_mean": t_vals.mean(), "never_mean": n_vals.mean(),
        "diff": diff, "pval": p_val,
    })

bal_tex = []
bal_tex.append(r"\begin{table}[ht!]")
bal_tex.append(r"\centering")
bal_tex.append(r"\caption{Baseline Balance: Treated vs.\ Never-Treated Counties (2009)}")
bal_tex.append(r"\label{tab:balance}")
bal_tex.append(r"\begin{threeparttable}")
bal_tex.append(r"\begin{tabular}{lcccc}")
bal_tex.append(r"\toprule")
bal_tex.append(r"Variable & Treated & Never-treated & Difference & $p$-value \\")
bal_tex.append(r"\midrule")
for r in bal_rows:
    s = stars(r["pval"])
    bal_tex.append(
        f"{r['label']} & {r['treated_mean']:.2f} & {r['never_mean']:.2f} & "
        f"{r['diff']:.2f}$^{{{s}}}$ & {r['pval']:.3f} \\\\"
    )
n_treated = int(baseline["ever_treated"].sum())
n_never = int((~baseline["ever_treated"].astype(bool)).sum())
bal_tex.append(r"\midrule")
bal_tex.append(f"Counties & {n_treated} & {n_never} & & \\\\")
bal_tex.append(r"\bottomrule")
bal_tex.append(r"\end{tabular}")
bal_tex.append(r"\begin{tablenotes}[flushleft]")
bal_tex.append(r"\item \small Two-sample $t$-tests for equality of means. Treated counties are those with any land coupon transaction during 2009--2020.")
bal_tex.append(r"\item $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$")
bal_tex.append(r"\end{tablenotes}")
bal_tex.append(r"\end{threeparttable}")
bal_tex.append(r"\end{table}")

bal_path = os.path.join(TAB_DIR, "table_balance.tex")
with open(bal_path, "w") as f:
    f.write("\n".join(bal_tex))
print(f"Balance table saved to {bal_path}")


# ############################################################################
# SECTION 7: INCOME MECHANISM TABLE (LaTeX)
# ############################################################################
print("\n" + "=" * 70)
print("7. INCOME MECHANISM TABLE (LaTeX — table_mechanism_income.tex)")
print("=" * 70)

income_results = []

# (1) Binary treatment → log rural income
m = pf.feols("ln_rural_income ~ treated | county_id + year",
             data=df, vcov={"CRV1": "county_id"})
income_results.append(("Rural", "Binary", m.coef()["treated"], m.se()["treated"],
                        m.pvalue()["treated"], len(df)))

# (2) Binary treatment → log urban income
m = pf.feols("ln_urban_income ~ treated | county_id + year",
             data=df, vcov={"CRV1": "county_id"})
income_results.append(("Urban", "Binary", m.coef()["treated"], m.se()["treated"],
                        m.pvalue()["treated"], len(df)))

# (3) Cumulative intensity → log rural income
m = pf.feols("ln_rural_income ~ ln_cum_area | county_id + year",
             data=df, vcov={"CRV1": "county_id"})
income_results.append(("Rural", "Cumulative", m.coef()["ln_cum_area"], m.se()["ln_cum_area"],
                        m.pvalue()["ln_cum_area"], len(df)))

# (4) Cumulative intensity → log urban income
m = pf.feols("ln_urban_income ~ ln_cum_area | county_id + year",
             data=df, vcov={"CRV1": "county_id"})
income_results.append(("Urban", "Cumulative", m.coef()["ln_cum_area"], m.se()["ln_cum_area"],
                        m.pvalue()["ln_cum_area"], len(df)))

for channel, ttype, est, se, p, n in income_results:
    print(f"  {channel} ({ttype}): est={est:.4f}, se={se:.4f}, p={p:.4f} {stars(p)}")

# Generate LaTeX table
inc_tex = []
inc_tex.append(r"\begin{table}[ht!]")
inc_tex.append(r"\centering")
inc_tex.append(r"\caption{Income Channel Decomposition: Pooled TWFE Estimates}")
inc_tex.append(r"\label{tab:income_channel}")
inc_tex.append(r"\begin{threeparttable}")
inc_tex.append(r"\begin{tabular}{lcccc}")
inc_tex.append(r"\toprule")
inc_tex.append(r"& \multicolumn{2}{c}{Binary Treatment} & \multicolumn{2}{c}{Cumulative Intensity} \\")
inc_tex.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
inc_tex.append(r"& ln(Rural inc.) & ln(Urban inc.) & ln(Rural inc.) & ln(Urban inc.) \\")
inc_tex.append(r"& (1) & (2) & (3) & (4) \\")
inc_tex.append(r"\midrule")

# Coefficients row
coef_line = "Treatment"
se_line = ""
for channel, ttype, est, se, p, n in income_results:
    coef_line += f" & {fmt_coef_tex(est, p)}"
    se_line += f" & {fmt_se_tex(se)}"
inc_tex.append(coef_line + r" \\")
inc_tex.append(se_line + r" \\")
inc_tex.append(r"\midrule")
inc_tex.append(r"County FE & Yes & Yes & Yes & Yes \\")
inc_tex.append(r"Year FE & Yes & Yes & Yes & Yes \\")
obs_line = "Observations"
for _, _, _, _, _, n in income_results:
    obs_line += f" & {n}"
inc_tex.append(obs_line + r" \\")
inc_tex.append(r"\bottomrule")
inc_tex.append(r"\end{tabular}")
inc_tex.append(r"\begin{tablenotes}[flushleft]")
inc_tex.append(r"\item \small Standard errors clustered by county in parentheses.")
inc_tex.append(r"\item $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$")
inc_tex.append(r"\end{tablenotes}")
inc_tex.append(r"\end{threeparttable}")
inc_tex.append(r"\end{table}")

inc_path = os.path.join(TAB_DIR, "table_mechanism_income.tex")
with open(inc_path, "w") as f:
    f.write("\n".join(inc_tex))
print(f"Income mechanism table saved to {inc_path}")


print("\n" + "=" * 70)
print("ROBUSTNESS AND MECHANISM ANALYSIS COMPLETE")
print("=" * 70)
