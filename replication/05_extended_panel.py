"""
============================================================================
05_extended_panel.py
============================================================================
Extended Panel Robustness Check (2005-2020)

This script replicates the extended-panel event study (Figure in paper):
  1. Loads the expanded county panel (2005-2020) with pre-2009 income data
  2. Constructs treatment variables (same logic as main analysis)
  3. Runs event-study DiD on the unbalanced extended panel
  4. Runs pooled TWFE on the extended panel
  5. Generates side-by-side comparison figure with balanced panel

Input:  data/Chongqing_expanded.xlsx
        data/county_panel.csv (for balanced panel comparison)
Output: output/figures/event_study_extended.png
        output/figures/event_study_extended.pdf
        output/event_study_extended.csv
============================================================================
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style, style_axis, plot_es, extract_es_coefs, C_BLUE, C_RED
import os
import warnings
warnings.filterwarnings("ignore")

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "output", "figures")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# 1. Load and prepare extended panel
# ============================================================================
print("=" * 70)
print("EXTENDED PANEL ROBUSTNESS CHECK (2005-2020)")
print("=" * 70)

df = pd.read_excel(os.path.join(DATA_DIR, "Chongqing_expanded.xlsx"))
df = df.dropna(subset=['urban_rural_income_ratio']).copy()
print(f"Extended panel: {len(df)} obs, {df['county_id'].nunique()} counties")
print(f"Year range: {df['year'].min()}-{df['year'].max()}")

# Construct treatment variables
first_treat = df[df['ticket_area'] > 0].groupby('county_id')['year'].min().reset_index()
first_treat.columns = ['county_id', 'first_treat_year']
df = df.merge(first_treat, on='county_id', how='left')
df['first_treat_year'] = df['first_treat_year'].fillna(0).astype(int)
df['treated'] = ((df['year'] >= df['first_treat_year']) & (df['first_treat_year'] > 0)).astype(int)
# Store original rel_time with NaN for never-treated
df['rel_time'] = np.where(df['first_treat_year'] > 0, df['year'] - df['first_treat_year'], np.nan)
df['ln_gdp_pc'] = np.log(df['gdp_pc'])
df['ln_fiscal_rev'] = np.log(df['fiscal_rev'])
df['ln_urban_rate'] = np.log(df['urbanization_rate'].clip(lower=0.01))
df['ln_cum_area'] = np.log(df['ticket_cum'] + 1)

# Report pre-2009 coverage
pre09 = df[df['year'] < 2009]
print(f"\nPre-2009 observations: {len(pre09)}")
print(f"Counties with pre-2009 data: {pre09['county_id'].nunique()}")
cohort_2009 = set(df[df['first_treat_year'] == 2009]['county_id'].unique())
pre09_in_2009cohort = pre09[pre09['county_id'].isin(cohort_2009)]
print(f"2009-cohort counties with pre-2009 data: {pre09_in_2009cohort['county_id'].nunique()}")


# ============================================================================
# 2. Event study on extended panel
# ============================================================================
print("\n" + "=" * 70)
print("EVENT STUDY ON EXTENDED PANEL")
print("=" * 70)


def extract_event_study(model, data):
    """Extract event-study coefficients from a pyfixest model using plot_style utility.

    Parameters
    ----------
    model : pyfixest model
        Fitted event-study model
    data : DataFrame
        Data used in model (must have 'rel_time' column with NaN for never-treated)

    Returns
    -------
    DataFrame with columns: rel_time, coef, se, ci_lo, ci_hi
    """
    # Get relative times and coefficients
    coefs_list = extract_es_coefs(model, data)

    # Convert to DataFrame
    rows = []
    for rt, coef, se in coefs_list:
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        rows.append({
            'rel_time': rt, 'coef': coef, 'se': se,
            'ci_lo': ci_lo, 'ci_hi': ci_hi
        })
    return pd.DataFrame(rows).sort_values('rel_time')


# Bin event time at [-4, +10] for model estimation
df_es = df.copy()
df_es['rt'] = df_es['rel_time'].copy()
df_es.loc[df_es['rt'] < -4, 'rt'] = -4
df_es.loc[df_es['rt'] > 10, 'rt'] = 10
df_es.loc[df_es['first_treat_year'] == 0, 'rt'] = -1.0

m_ext = pf.feols("urban_rural_income_ratio ~ i(rt, ref=-1.0) | county_id + year",
                 data=df_es, vcov={"CRV1": "county_id"})
ext_es = extract_event_study(m_ext, df_es)

print("\nExtended panel event-study coefficients:")
for _, r in ext_es.iterrows():
    t = abs(r['coef'] / max(r['se'], 0.001))
    star = "***" if t > 2.576 else "**" if t > 1.96 else "*" if t > 1.645 else ""
    print(f"  e={int(r['rel_time']):+3d}: coef={r['coef']:+.4f} (SE={r['se']:.4f}) {star}")

# Save results
ext_es.to_csv(os.path.join(OUT_DIR, "event_study_extended.csv"), index=False)
print(f"\nResults saved to {os.path.join(OUT_DIR, 'event_study_extended.csv')}")


# ============================================================================
# 3. Pooled TWFE on extended panel
# ============================================================================
print("\n" + "=" * 70)
print("POOLED TWFE ON EXTENDED PANEL")
print("=" * 70)

m1 = pf.feols("urban_rural_income_ratio ~ treated | county_id + year",
              data=df, vcov={"CRV1": "county_id"})
print(f"Binary, no ctrl: coef={m1.coef().iloc[0]:.4f}, se={m1.se().iloc[0]:.4f}, "
      f"p={m1.pvalue().iloc[0]:.4f}, N={m1._N}, R2w={m1._r2_within:.4f}")

df_ctrl = df.dropna(subset=['ln_urban_rate'])
m2 = pf.feols("urban_rural_income_ratio ~ treated + ln_gdp_pc + ln_fiscal_rev + ln_urban_rate | county_id + year",
              data=df_ctrl, vcov={"CRV1": "county_id"})
print(f"Binary, +ctrl: coef={m2.coef().iloc[0]:.4f}, se={m2.se().iloc[0]:.4f}, "
      f"p={m2.pvalue().iloc[0]:.4f}, N={m2._N}")

m3 = pf.feols("urban_rural_income_ratio ~ ln_cum_area | county_id + year",
              data=df, vcov={"CRV1": "county_id"})
print(f"Cumulative, no ctrl: coef={m3.coef().iloc[0]:.4f}, se={m3.se().iloc[0]:.4f}, "
      f"p={m3.pvalue().iloc[0]:.4f}, N={m3._N}")


# ============================================================================
# 4. Generate comparison figure
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING COMPARISON FIGURE")
print("=" * 70)

# Load balanced panel for comparison
orig = pd.read_csv(os.path.join(DATA_DIR, "county_panel.csv"))
# Fix rel_time: replace -1 with NaN for never-treated
orig['rel_time'] = orig['rel_time'].replace(-1, np.nan)

orig_es_data = orig.copy()
orig_es_data['rt'] = orig_es_data['rel_time'].copy()
orig_es_data.loc[orig_es_data['rt'] < -4, 'rt'] = -4
orig_es_data.loc[orig_es_data['rt'] > 10, 'rt'] = 10
orig_es_data.loc[orig_es_data['first_treat_year'] == 0, 'rt'] = -1.0

m_orig = pf.feols("urban_rural_income_ratio ~ i(rt, ref=-1.0) | county_id + year",
                   data=orig_es_data, vcov={"CRV1": "county_id"})
orig_es = extract_event_study(m_orig, orig_es_data)

# Apply unified style
apply_style()

# Create side-by-side figure with standardized formatting
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a): Balanced panel
plot_es(axes[0], orig_es['rel_time'], orig_es['coef'], orig_es['se'],
        color=C_BLUE, marker='o', band=True)
axes[0].set_title('(a) Balanced panel, 2009\u20132020\n($N$ = 444, 37 counties)')
axes[0].set_xlim(-4.5, 10.5)
axes[0].set_xticks(range(-4, 11, 2))
axes[0].set_ylabel('Coefficient (Urban\u2013Rural Income Ratio)')
style_axis(axes[0], ylabel='Coefficient (Urban\u2013Rural Income Ratio)')

# Panel (b): Extended panel
plot_es(axes[1], ext_es['rel_time'], ext_es['coef'], ext_es['se'],
        color=C_RED, marker='s', band=True)
axes[1].set_title('(b) Extended panel, 2005\u20132020\n($N$ = 504, 37 counties, unbalanced)')
axes[1].set_xlim(-4.5, 10.5)
axes[1].set_xticks(range(-4, 11, 2))
style_axis(axes[1])

# Share y-axis
axes[1].set_ylim(axes[0].get_ylim())

plt.tight_layout()
for ext in ['png', 'pdf']:
    plt.savefig(os.path.join(FIG_DIR, f"event_study_extended.{ext}"),
                dpi=300, bbox_inches='tight')
plt.close()
print("Figure saved: event_study_extended.png / .pdf")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
