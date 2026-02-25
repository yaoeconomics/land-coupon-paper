"""
============================================================================
07_national_figure.py
============================================================================
National Urban-Rural Income Ratio Figure (Figure 1)

Generates Figure 1 in the paper: a bar chart of national urban and rural
per capita disposable income with an overlaid line for the income ratio,
marking the 2008 launch of the Land Coupon program.

Input:  china_urban_rural_income_2005_2020.xlsx
Output: output/figures/China_urban_rural_ratio_figure.png
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from plot_style import apply_style, C_BLUE, C_ORANGE, C_RED

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "china_urban_rural_income_2005_2020.xlsx")
FIG_DIR = os.path.join(BASE_DIR, "output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# Load data
# ============================================================================
apply_style()

df = pd.read_excel(DATA_FILE)
print(f"Loaded national income data: {df.shape}")
print(df.head())

# Identify columns — expect Year, Urban income, Rural income
# Try common column name patterns
cols = df.columns.tolist()
print(f"Columns: {cols}")

# Attempt to auto-detect or use positional
# Typical structure: Year | Urban income | Rural income
year_col = [c for c in cols if 'year' in c.lower() or 'Year' in c]
if year_col:
    year_col = year_col[0]
else:
    year_col = cols[0]

urban_col = [c for c in cols if 'urban' in c.lower() and 'rural' not in c.lower() and 'ratio' not in c.lower()]
rural_col = [c for c in cols if 'rural' in c.lower() and 'ratio' not in c.lower()]

if urban_col and rural_col:
    urban_col = urban_col[0]
    rural_col = rural_col[0]
else:
    # Fallback: assume cols[1]=urban, cols[2]=rural
    urban_col = cols[1]
    rural_col = cols[2]

df['year'] = df[year_col].astype(int)
df['urban_income'] = pd.to_numeric(df[urban_col], errors='coerce')
df['rural_income'] = pd.to_numeric(df[rural_col], errors='coerce')
df['ratio'] = df['urban_income'] / df['rural_income']

# Filter to 2005-2020
df = df[(df['year'] >= 2005) & (df['year'] <= 2020)].sort_values('year').reset_index(drop=True)

print(f"\nYear range: {df['year'].min()}–{df['year'].max()}")
print(f"Income ratio range: {df['ratio'].min():.2f}–{df['ratio'].max():.2f}")

# ============================================================================
# Generate Figure
# ============================================================================
fig, ax1 = plt.subplots(figsize=(8, 4.5))

bar_width = 0.35
x = np.arange(len(df))

# Bar chart for income levels (left axis, in thousands)
bars_urban = ax1.bar(x - bar_width / 2, df['urban_income'] / 1000, bar_width,
                      color=C_BLUE, alpha=0.75, label='Urban income')
bars_rural = ax1.bar(x + bar_width / 2, df['rural_income'] / 1000, bar_width,
                      color=C_ORANGE, alpha=0.75, label='Rural income')

ax1.set_ylabel('Per capita disposable income (thousand RMB)')
ax1.set_xlabel('')
ax1.set_xticks(x)
ax1.set_xticklabels(df['year'].astype(str), rotation=45, ha='right')
ax1.spines['top'].set_visible(False)

# Income ratio (right axis)
ax2 = ax1.twinx()
ax2.plot(x, df['ratio'], color=C_RED, marker='o', markersize=4, linewidth=1.5,
         label='Income ratio', zorder=5)
ax2.set_ylabel('Urban–rural income ratio')
ax2.spines['top'].set_visible(False)

# Mark LCP launch (2008)
lcp_x = list(df['year']).index(2008) if 2008 in df['year'].values else None
if lcp_x is not None:
    ax1.axvline(x=lcp_x, color=C_RED, linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
    ax1.text(lcp_x + 0.1, ax1.get_ylim()[1] * 0.95, 'LCP launch\n(2008)',
             fontsize=8, color=C_RED, va='top')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, frameon=False)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "China_urban_rural_ratio_figure.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, "China_urban_rural_ratio_figure.pdf"), bbox_inches='tight')
plt.close()

print("\nNational income ratio figure saved.")
