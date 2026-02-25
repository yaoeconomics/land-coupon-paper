"""
============================================================================
01_data_preparation.py
============================================================================
Data Preparation for Staggered DiD Analysis of China's Land Coupon Program

This script:
  1. Loads the county-level panel data from Chongqing.xlsx
  2. Identifies treatment cohorts (year of first land coupon transaction)
  3. Constructs analysis variables (relative time, treatment indicators, etc.)
  4. Reports descriptive statistics and panel diagnostics
  5. Saves the prepared dataset for subsequent analysis

Input:  data/Chongqing.xlsx
Output: data/county_panel.csv
============================================================================
"""

import pandas as pd
import numpy as np
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "Chongqing.xlsx")
OUTPUT_FILE = os.path.join(DATA_DIR, "county_panel.csv")

# ============================================================================
# 1. Load raw data
# ============================================================================
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

df = pd.read_excel(INPUT_FILE)
print(f"Raw data: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Years: {sorted(df['year'].unique())}")
print(f"Counties: {df['county_id'].nunique()} unique")

# ============================================================================
# 2. Identify treatment cohorts
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: IDENTIFYING TREATMENT COHORTS")
print("=" * 70)

# A county is "treated" starting from the first year it has a positive
# land coupon transaction area (ticket_area > 0).
first_transaction = (
    df[df["ticket_area"] > 0]
    .groupby("county_id")["year"]
    .min()
    .reset_index()
    .rename(columns={"year": "first_treat_year"})
)

# Merge back; never-treated counties receive first_treat_year = 0
df = df.merge(first_transaction, on="county_id", how="left")
df["first_treat_year"] = df["first_treat_year"].fillna(0).astype(int)

# Report cohort distribution
n_treated = (df["first_treat_year"] > 0).groupby(df["county_id"]).first().sum()
n_never = df["county_id"].nunique() - n_treated
print(f"Counties with ≥1 transaction: {n_treated}")
print(f"Never-treated counties: {n_never}")

cohort_counts = df.groupby("first_treat_year")["county_id"].nunique()
print("\nCohort distribution:")
for yr, n in cohort_counts.items():
    label = "Never treated" if yr == 0 else f"First treated {yr}"
    print(f"  {label}: {n} counties")

# ============================================================================
# 3. Construct analysis variables
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: CONSTRUCTING ANALYSIS VARIABLES")
print("=" * 70)

# Binary treatment indicator: 1 if county i is treated at time t
df["treated"] = np.where(
    (df["first_treat_year"] > 0) & (df["year"] >= df["first_treat_year"]), 1, 0
)

# Relative time (event time): year - first_treat_year; NaN for never-treated
df["rel_time"] = np.where(
    df["first_treat_year"] > 0,
    df["year"] - df["first_treat_year"],
    np.nan,
)

# Log-transformed controls
df["ln_gdp_pc"] = np.log(df["gdp_pc"])
df["ln_fiscal_rev"] = np.log(df["fiscal_rev"])
df["ln_urban_rate"] = np.log(df["urbanization_rate"])

# Log income components (for mechanism analysis)
df["ln_rural_income"] = np.log(df["mu_r"])
df["ln_urban_income"] = np.log(df["mu_u"])

# Log treatment intensity variables
df["ln_annual_area"] = np.log(df["ticket_area"] + 1)
df["ln_cum_area"] = np.log(df["ticket_cum"] + 1)

print("Variables constructed: treated, rel_time, ln_gdp_pc, ln_fiscal_rev,")
print("  ln_urban_rate, ln_rural_income, ln_urban_income, ln_annual_area, ln_cum_area")

# ============================================================================
# 4. Panel diagnostics
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: PANEL DIAGNOSTICS")
print("=" * 70)

obs_per_county = df.groupby("county_id")["year"].count()
print(f"Obs per county: min={obs_per_county.min()}, max={obs_per_county.max()}")
print(f"Balanced panel: {obs_per_county.min() == obs_per_county.max()}")
print(f"Total observations: {len(df)}")
print(f"Treatment obs: {df['treated'].sum()} ({df['treated'].mean()*100:.1f}%)")

# ============================================================================
# 5. Baseline (2009) characteristics by treatment status
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: BASELINE CHARACTERISTICS (2009)")
print("=" * 70)

baseline = df[df["year"] == 2009].copy()
baseline["ever_treated"] = (baseline["first_treat_year"] > 0).astype(int)

for var, label in [
    ("urban_rural_income_ratio", "Urban-rural income ratio"),
    ("gdp_pc", "GDP per capita (RMB)"),
    ("urbanization_rate", "Urbanization rate"),
]:
    treated_vals = baseline.loc[baseline["ever_treated"] == 1, var]
    never_vals = baseline.loc[baseline["ever_treated"] == 0, var]
    print(f"\n{label}:")
    print(f"  Treated (n={len(treated_vals)}): mean={treated_vals.mean():.2f}")
    print(f"  Never-treated (n={len(never_vals)}): mean={never_vals.mean():.2f}")

# ============================================================================
# 6. Save prepared dataset
# ============================================================================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n{'=' * 70}")
print(f"Prepared dataset saved to: {OUTPUT_FILE}")
print(f"  {df.shape[0]} rows × {df.shape[1]} columns")
print(f"{'=' * 70}")
