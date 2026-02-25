"""
06_rural_income.py
==================
Comprehensive event-study analysis of rural disposable income.

Produces:
    - event_study_rural_income.png / .pdf  (3-panel figure)
    - table_rural_income.tex               (pooled TWFE + CS overall ATT)

Dependencies:
    - pyfixest >= 0.25
    - csdid == 0.3.0  (pin for reproducibility)
    - drdid == 0.1.0
    - matplotlib >= 3.7
    - pandas, numpy, openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import pyfixest as pf
from csdid.att_gt import ATTgt
from plot_style import (apply_style, style_axis, plot_es, extract_es_coefs,
                         fmt_coef_tex, fmt_se_tex, C_BLUE, C_RED, C_GREEN, stars)

# ── paths ────────────────────────────────────────────────────
DATA_BAL  = 'data/county_panel.csv'
DATA_EXT  = 'data/Chongqing_expanded.xlsx'
OUT_DIR   = '../overleaf_files/'

# ── (a) load balanced panel ─────────────────────────────────
df = pd.read_csv(DATA_BAL)
df_twfe = df.copy()
df_twfe['rel_time'] = df_twfe['rel_time'].fillna(-99)

# ── (b) load & prepare extended panel ───────────────────────
dfx = pd.read_excel(DATA_EXT)
dfx['ln_rural_income'] = np.log(dfx['mu_r'])
dfx['treated'] = (dfx['ticket_area'] > 0).astype(int)
for cid in dfx['county_id'].unique():
    mask = (dfx['county_id'] == cid) & (dfx['ticket_area'] > 0)
    if mask.any():
        dfx.loc[dfx['county_id'] == cid, 'first_treat_year'] = dfx.loc[mask, 'year'].min()
    else:
        dfx.loc[dfx['county_id'] == cid, 'first_treat_year'] = 0
dfx['first_treat_year'] = dfx['first_treat_year'].astype(int)
dfx['rel_time'] = np.where(dfx['first_treat_year'] > 0,
                            dfx['year'] - dfx['first_treat_year'], np.nan)
dfx_ri = dfx.dropna(subset=['ln_rural_income']).copy()
dfx_twfe = dfx_ri.copy()
dfx_twfe['rel_time'] = dfx_twfe['rel_time'].fillna(-99)

print(f"Balanced panel: N={len(df)}, counties={df['county_id'].nunique()}")
print(f"Extended panel:  N={len(dfx_ri)}, counties={dfx_ri['county_id'].nunique()}")

# ══════════════════════════════════════════════════════════════
# 1. TWFE event studies
# ══════════════════════════════════════════════════════════════
# Balanced panel TWFE
mod_bal = pf.feols(
    'ln_rural_income ~ i(rel_time, ref=-1.0) | county_id + year',
    data=df_twfe, vcov={'CRV1': 'county_id'}
)
coefs_bal = extract_es_coefs(mod_bal, df)

# Extended panel TWFE
mod_ext = pf.feols(
    'ln_rural_income ~ i(rel_time, ref=-1.0) | county_id + year',
    data=dfx_twfe, vcov={'CRV1': 'county_id'}
)
coefs_ext = extract_es_coefs(mod_ext, dfx_ri)

# ══════════════════════════════════════════════════════════════
# 2. Callaway–Sant'Anna doubly robust
# ══════════════════════════════════════════════════════════════
cs = ATTgt(
    yname='ln_rural_income', tname='year', idname='county_id',
    gname='first_treat_year', data=df, control_group='nevertreated',
    panel=True, biters=999, alp=0.05
)
result = cs.fit(est_method='dr')
es_cs  = result.aggte(typec='dynamic')

cs_egt = es_cs.atte['egt']
cs_att = es_cs.atte['att_egt']
cs_se  = es_cs.atte['se_egt'].flatten()

print(f"\nCS overall ATT: {es_cs.atte['overall_att']:.4f} "
      f"(SE={es_cs.atte['overall_se'][0]:.4f})")

# ══════════════════════════════════════════════════════════════
# 3. Pooled TWFE regressions
# ══════════════════════════════════════════════════════════════
mod_bin_bal = pf.feols('ln_rural_income ~ treated | county_id + year',
                       data=df, vcov={'CRV1': 'county_id'})
mod_cum_bal = pf.feols('ln_rural_income ~ ln_cum_area | county_id + year',
                       data=df, vcov={'CRV1': 'county_id'})
mod_bin_ext = pf.feols('ln_rural_income ~ treated | county_id + year',
                       data=dfx_ri, vcov={'CRV1': 'county_id'})

print(f"\nBalanced binary:     {mod_bin_bal.coef().iloc[0]:.4f} (p={mod_bin_bal.pvalue().iloc[0]:.4f})")
print(f"Balanced cumulative: {mod_cum_bal.coef().iloc[0]:.4f} (p={mod_cum_bal.pvalue().iloc[0]:.4f})")
print(f"Extended binary:     {mod_bin_ext.coef().iloc[0]:.4f} (p={mod_bin_ext.pvalue().iloc[0]:.4f})")

# ══════════════════════════════════════════════════════════════
# 4. Three-panel figure
# ══════════════════════════════════════════════════════════════
apply_style()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

bal_rt  = [c[0] for c in coefs_bal]
bal_c   = [c[1] for c in coefs_bal]
bal_s   = [c[2] for c in coefs_bal]

ext_rt  = [c[0] for c in coefs_ext]
ext_c   = [c[1] for c in coefs_ext]
ext_s   = [c[2] for c in coefs_ext]

# Panel (a): TWFE balanced
plot_es(axes[0], bal_rt, bal_c, bal_s, C_BLUE, 'o', band=True)
axes[0].set_title(r'(a) TWFE, balanced panel' + '\n' + r'($N$ = 444)')
axes[0].set_ylabel('Coefficient (log rural income)')
style_axis(axes[0])

# Panel (b): CS DR
plot_es(axes[1], cs_egt, cs_att, list(cs_se), C_RED, 's', band=True)
axes[1].set_title(r'(b) Callaway–Sant\'Anna DR' + '\n' + r'($N_{\mathrm{eff}}$ = 276)')
style_axis(axes[1])

# Panel (c): TWFE extended
plot_es(axes[2], ext_rt, ext_c, ext_s, C_GREEN, '^', band=True)
axes[2].set_title(r'(c) TWFE, extended panel' + '\n' + f'($N$ = {len(dfx_ri)})')
style_axis(axes[2])

# Unified y-limits across all panels
y_min = -0.08
y_max = 0.25
for ax in axes:
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Event time (years since adoption)')

plt.tight_layout()
plt.savefig(OUT_DIR + 'event_study_rural_income.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR + 'event_study_rural_income.pdf', bbox_inches='tight')
print("\nFigure saved to", OUT_DIR)

# ══════════════════════════════════════════════════════════════
# 5. LaTeX table
# ══════════════════════════════════════════════════════════════
cs_att_val = es_cs.atte['overall_att']
cs_se_val  = es_cs.atte['overall_se'][0]
cs_p       = 2*(1 - __import__('scipy').stats.norm.cdf(abs(cs_att_val/cs_se_val)))

tex = r"""\begin{table}[htbp]
\centering
\caption{Effect of Land Coupon Program on Rural Disposable Income}
\label{tab:rural_income}
\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{4}{c}{Dependent variable: Log rural disposable income} \\
\midrule
 & (1) & (2) & (3) & (4) \\
 & Binary & Cumulative & Extended & CS--DR \\
 & TWFE & TWFE & Panel TWFE & Overall ATT \\
\midrule
"""
# Row 1: binary treatment
coef_1 = fmt_coef_tex(mod_bin_bal.coef().iloc[0], mod_bin_bal.pvalue().iloc[0])
se_1 = fmt_se_tex(mod_bin_bal.se().iloc[0])
coef_3 = fmt_coef_tex(mod_bin_ext.coef().iloc[0], mod_bin_ext.pvalue().iloc[0])
se_3 = fmt_se_tex(mod_bin_ext.se().iloc[0])
tex += f"Treated (binary) & {coef_1} & & {coef_3} & \\\\\n"
tex += f" & {se_1} & & {se_3} & \\\\[3pt]\n"
# Row 2: cumulative
coef_2 = fmt_coef_tex(mod_cum_bal.coef().iloc[0], mod_cum_bal.pvalue().iloc[0])
se_2 = fmt_se_tex(mod_cum_bal.se().iloc[0])
tex += f"Cumulative area (log) & & {coef_2} & & \\\\\n"
tex += f" & & {se_2} & & \\\\[3pt]\n"
# Row 3: CS
coef_cs = fmt_coef_tex(cs_att_val, cs_p)
se_cs = fmt_se_tex(cs_se_val)
tex += f"CS overall ATT & & & & {coef_cs} \\\\\n"
tex += f" & & & & {se_cs} \\\\[6pt]\n"
tex += r"""\midrule
County FE & Yes & Yes & Yes & --- \\
Year FE & Yes & Yes & Yes & --- \\
"""
tex += f"Panel & 2009--2020 & 2009--2020 & 2005--2020 & 2009--2020 \\\\\n"
tex += f"Observations & {mod_bin_bal._N} & {mod_cum_bal._N} & {mod_bin_ext._N} & 276 \\\\\n"
tex += f"Clusters & 37 & 37 & 37 & 23 \\\\\n"
tex += f"$R^2$ (within) & {mod_bin_bal._r2_within:.3f} & {mod_cum_bal._r2_within:.3f} & {mod_bin_ext._r2_within:.3f} & --- \\\\\n"
tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Standard errors clustered at the county level in parentheses.
Column~(1) reports the binary TWFE estimate on the balanced 2009--2020 panel.
Column~(2) uses cumulative transaction area (log) as a continuous treatment intensity measure.
Column~(3) extends the panel to 2005--2020.
Column~(4) reports the Callaway and Sant'Anna (2021) doubly robust overall ATT;
the 2009 cohort is dropped because it lacks a pre-treatment period.
*** $p<0.01$, ** $p<0.05$, * $p<0.1$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""

with open(OUT_DIR + 'table_rural_income.tex', 'w') as f:
    f.write(tex)
print("Table saved to", OUT_DIR + 'table_rural_income.tex')
print("\nDone.")
