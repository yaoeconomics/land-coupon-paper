"""
plot_style.py
=============
Shared plotting style and helper functions for all replication scripts.
Ensures visual consistency across all figures in the paper.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re

# ── Unified rcParams ─────────────────────────────────────────
RCPARAMS = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
}

# ── Color palette ────────────────────────────────────────────
C_BLUE   = '#2166ac'
C_RED    = '#b2182b'
C_GREEN  = '#1b7837'
C_ORANGE = '#e66101'
C_GRAY   = '#636363'


def apply_style():
    """Apply unified matplotlib style to all subsequent plots."""
    plt.rcParams.update(RCPARAMS)


def style_axis(ax, ylabel=None, xlabel='Years relative to first transaction'):
    """Apply consistent styling to an axis."""
    ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
    ax.axvline(x=-0.5, color=C_GRAY, linewidth=0.7, linestyle='--', alpha=0.7, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(direction='out', length=3)


def plot_es(ax, rts, coefs, ses, color, marker='o', label=None, band=True):
    """Plot event-study coefficients with CI band and error bars."""
    coefs = np.array(coefs, dtype=float)
    ses = np.array(ses, dtype=float)
    ci_lo = coefs - 1.96 * ses
    ci_hi = coefs + 1.96 * ses
    if band:
        ax.fill_between(rts, ci_lo, ci_hi, alpha=0.15, color=color, linewidth=0)
    ax.errorbar(rts, coefs, yerr=1.96 * ses,
                fmt=f'{marker}-', color=color, markersize=4, linewidth=1.2,
                capsize=1.5, capthick=0.7, markeredgewidth=0.5,
                label=label, zorder=3)


def extract_es_coefs(mod, data_raw):
    """Extract event-study coefficients from a pyfixest model.

    Robust to different pyfixest coefficient naming formats.
    Returns list of (rel_time, coef, se) tuples.
    """
    valid_rt = sorted([r for r in data_raw['rel_time'].dropna().unique()])
    coef_names = list(mod._coefnames)
    coef_vals = mod.coef()
    se_vals = mod.se()

    name_map = {}
    for i, name in enumerate(coef_names):
        m = re.search(r'\[T\.([-\d.]+)\]', str(name))
        if not m:
            m = re.search(r'C\(rel_time,\s*([-\d.]+)\)', str(name))
        if m:
            name_map[float(m.group(1))] = i

    out = []
    for k in valid_rt:
        k_int = int(k)
        if k_int == -1:
            out.append((k_int, 0.0, 0.0))
            continue
        if k in name_map:
            idx = name_map[k]
            out.append((k_int, float(coef_vals.iloc[idx]), float(se_vals.iloc[idx])))
    return out


def stars(p):
    """Return significance stars for a p-value."""
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


def fmt_coef_tex(est, p):
    """Format coefficient with significance stars for LaTeX."""
    return f"${est:.3f}^{{{stars(p)}}}$"


def fmt_se_tex(se):
    """Format standard error for LaTeX."""
    return f"$({se:.3f})$"
