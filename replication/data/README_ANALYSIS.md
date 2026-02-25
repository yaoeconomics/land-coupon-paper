# Land Coupon Paper: Dataset Analysis Report
**Date:** February 24, 2026

## Executive Summary

This analysis covers two complementary datasets on the "Land Coupon" (土地券) policy in China:
1. **Chengdu Data** (成都): 20 counties × 16 years (2005-2020) = 320 observations
2. **Chongqing Data** (重庆): 37 counties × 12 years (2009-2020) = 444 observations

Both datasets have **perfectly balanced panels** and contain treatment variables for the land coupon policy, making them suitable for difference-in-differences (DiD) estimation.

## Key Findings

### Chongqing: Never-Treated Counties Identified (6 counties, 16.2%)

The following 6 counties **never adopted land coupons** (land_coupon_transaction_binary = 0 throughout 2009-2020):

1. **南岸区** (Nan'an District)
2. **大渡口区** (Dadukou District)  
3. **大足区** (Dazu District)
4. **江北区** (Jiangbei District)
5. **沙坪坝区** (Shapingba District)
6. **长寿区** (Changshou District)

The remaining **31 counties** adopted the policy at various times.

### Critical Discovery: Severe Selection Bias in Chongqing

**Never-treated counties are SYSTEMATICALLY RICHER and MORE URBANIZED:**

| Characteristic | Never-Treated (n=6) | Treated (n=31) | Difference |
|---|---|---|---|
| Urban Income (mu_u) | 16,440 | 13,860 | +18.6% (NT richer) |
| Rural Income (mu_r) | 6,912 | 4,703 | +47.0% (NT richer) |
| Income Ratio (u/r) | 2.408 | 3.027 | -20.4% (T has more inequality) |
| GDP per Capita | 36,761 | 18,401 | **+99.9% (NT much richer!)** |
| Urbanization Rate | 74.23% | 40.46% | +83.4% (NT more urban) |
| Population (10k) | 66.33 | 89.93 | -26.3% (T larger) |

**Interpretation:** The policy targets poorer, less urbanized areas. This is **selection on observables** - treatment assignment is NOT random. Standard difference-in-differences estimators may be biased.

## Dataset Specifications

### Chengdu Dataset

**File:** `Chengdu_counties_data.xlsx`

**Dimensions:**
- 320 observations (BALANCED: 20 counties × 16 years)
- 43 variables
- Time period: 2005-2020

**County Names (20):**
双流区, 大邑县, 崇州市, 彭州市, 成华区, 新津县, 新都区, 武侯区, 温江区, 简阳市, 蒲江县, 邛崃市, 郫都区, 都江堰市, 金堂县, 金牛区, 锦江区, 青白江区, 青羊区, 龙泉驿区

**Data Quality:** MODERATE
- Excellent: GDP data, treatment indicators, schools
- Good: Population (86.9%), employment (68.8%)
- **Critical Issue:** Urbanization rate is 100% MISSING
- **Missing:** mu_u (59.4%), mu_r (70.6%)

**Key Variables:**
```
✓ mu_u (urban income) - 59.4% complete
✓ mu_r (rural income) - 70.6% complete
✗ income_ratio - MUST COMPUTE as mu_u / mu_r
✓ gdp_pc (GDP per capita) - 100% complete
✗ urbanization_rate - 0% complete (NO DATA)
✓ total_pop_registered - 86.9% complete
✓ land_coupon_transaction_binary - 100% complete
✓ after_first_land_coupon_transaction - 100% complete
```

### Chongqing Dataset

**File:** `Chongqing.xlsx`

**Dimensions:**
- 444 observations (BALANCED: 37 counties × 12 years)
- 48 variables  
- Time period: 2009-2020

**County Names (37):**
万州区, 丰都县, 九龙坡区, 云阳县, 北碚区, 南岸区, 南川区, 合川区, 垫江县, 城口县, 大渡口区, 大足区, 奉节县, 巫山县, 巫溪县, 巴南区, 开州区, 彭水苗族土家族自治县, 忠县, 梁平区, 武隆区, 永川区, 江北区, 江津区, 沙坪坝区, 涪陵区, 渝北区, 潼南区, 璧山区, 石柱土家族自治县, 秀山土家族苗族自治县, 綦江区, 荣昌区, 酉阳土家族苗族自治县, 铜梁区, 长寿区, 黔江区

**Data Quality:** EXCELLENT
- 95%+ complete for all core variables
- Minimal missing data: only industrial enterprises (80.2%), oilseed output (94.6%)

**Key Variables:**
```
✓ mu_u (urban income) - 100% complete
✓ mu_r (rural income) - 100% complete
✓ urban_rural_income_ratio - 100% complete (pre-calculated!)
✓ gdp_pc (GDP per capita) - 100% complete
✓ GDP_total - 100% complete
✓ urbanization_rate - 100% complete
✓ total_pop_registered - 100% complete
✓ land_coupon_transaction_binary - 100% complete
✓ after_first_land_coupon_transaction - 100% complete
```

## Summary Statistics

### Chengdu (2005-2020)

| Variable | n | Mean | Std Dev | Min | Max |
|---|---|---|---|---|---|
| Urban Income (mu_u) | 190 | 32,346 | 9,820 | 11,825 | 52,036 |
| Rural Income (mu_r) | 226 | 13,490 | 8,421 | 3,092 | 51,299 |
| GDP per Capita | 320 | 52,997 | 33,879 | 6,223 | 161,656 |
| Population (10k) | 278 | 67.0 | 29.8 | 25.7 | 207.0 |

### Chongqing (2009-2020)

| Variable | n | Mean | Std Dev | Min | Max |
|---|---|---|---|---|---|
| Urban Income (mu_u) | 444 | 26,171 | 8,564 | 10,323 | 46,429 |
| Rural Income (mu_r) | 444 | 11,228 | 4,852 | 3,078 | 24,869 |
| Income Ratio | 444 | 2.465 | 0.390 | 1.784 | 3.802 |
| GDP per Capita | 444 | 44,096 | 24,079 | 7,056 | 143,380 |
| Urbanization Rate (%) | 444 | 55.4 | 20.4 | 21.6 | 99.2 |
| Population (10k) | 444 | 89.3 | 35.4 | 23.3 | 176.1 |

## Variable Comparability

| Variable | Chengdu | Chongqing | Compatible |
|---|---|---|---|
| Urban Income | ✓ (59.4%) | ✓ (100%) | YES |
| Rural Income | ✓ (70.6%) | ✓ (100%) | YES |
| Income Ratio | ✗ COMPUTE | ✓ (100%) | DIFFERENT |
| GDP per Capita | ✓ (100%) | ✓ (100%) | YES |
| Urbanization Rate | ✗ (0% NO DATA) | ✓ (100%) | DIFFERENT |
| Population | ✓ (86.9%) | ✓ (100%) | YES |
| Land Coupon Binary | ✓ (100%) | ✓ (100%) | YES |
| After First Coupon | ✓ (100%) | ✓ (100%) | YES |

## Baseline (2009) Characteristics by Treatment Status - Chongqing

### Never-Treated Counties (n=6)

| County | Urban Income | Rural Income | Ratio | GDP pc | Urban % | Pop (10k) |
|---|---|---|---|---|---|---|
| 南岸区 | 17,210 | 7,955 | 2.16 | 42,307 | 89.16 | 58.37 |
| 大渡口区 | 17,183 | 7,612 | 2.26 | 54,446 | 92.76 | 23.27 |
| 大足区 | 14,697 | 5,604 | 2.62 | 15,432 | 36.09 | 96.00 |
| 江北区 | 17,263 | 7,444 | 2.32 | 47,051 | 88.63 | 53.54 |
| 沙坪坝区 | 17,361 | 7,421 | 2.34 | 38,286 | 88.22 | 76.70 |
| 长寿区 | 14,927 | 5,437 | 2.75 | 23,043 | 50.55 | 90.10 |
| **Mean** | **16,440** | **6,912** | **2.41** | **36,761** | **74.24%** | **66.33** |
| **Std** | 1,265 | 1,096 | 0.23 | 14,798 | 24.43% | 26.96 |

### Treated Counties (n=31) - Summary

| Statistic | Urban Income | Rural Income | Ratio | GDP pc | Urban % | Pop (10k) |
|---|---|---|---|---|---|---|
| **Mean** | 13,860 | 4,703 | 3.027 | 18,401 | 40.46% | 89.93 |
| **Std** | 2,022 | 1,156 | 0.384 | 10,561 | 17.01% | 35.58 |
| **Min** | 10,323 | 3,078 | 1.784 | 7,056 | 21.60% | 24.37 |
| **Max** | 17,210 | 7,440 | 3.605 | 49,916 | 84.11% | 172.82 |

## Methodological Issues & Recommendations

### Chongqing: Selection Bias

**Problem:** Never-treated and treated groups differ systematically in pre-treatment characteristics.

**Solutions:**
1. **Propensity Score Matching** - Create balanced treated/control subsamples
2. **Inverse Probability Weighting** - Weight observations by P(treatment | X)
3. **Covariate Balancing** - Use weights that balance covariates
4. **Control Variable Approach** - Include baseline X in DiD regression
5. **Parallel Trends Check** - Verify trends are similar in pre-period

### Chengdu: Missing Data

**Problems:**
1. Urbanization rate completely missing (100%)
2. Income variables missing for ~40% of observations
3. Sample size reduction if using complete cases

**Solutions:**
1. **Drop** 城镇化率_常住人口 variable (unusable)
2. **COMPUTE** income_ratio = mu_u / mu_r
3. **Subsample** to observations with valid mu_u and mu_r
4. **Document** sample size reduction in final analysis
5. **Consider** multiple imputation if missing data is not MCAR

### Cross-Regional Comparison

**Feasibility Issues:**
- Different time periods (can overlap only 2009-2020)
- Different data quality (Chongqing much cleaner)
- May have different treatment effects (regional heterogeneity)
- Different urbanization contexts

**Recommendation:** Analyze separately, then compare effect sizes with caution.

## Data Preparation Checklist

### For Chongqing Analysis

- [ ] Subset to 2009-2020 period
- [ ] Identify treatment timing for each county (when did they first use coupons?)
- [ ] Create treatment indicator (D = 1 if adopted, 0 if never)
- [ ] Create year dummies for fixed effects
- [ ] Create county dummies for fixed effects
- [ ] Check for parallel trends (plot pre-period trends by group)
- [ ] Decide on matching/weighting strategy for selection bias
- [ ] Winsorize outliers if necessary (check GDP pc distribution)
- [ ] Create interaction terms for DiD (treat × post)

### For Chengdu Analysis

- [ ] **COMPUTE** income_ratio = mu_u / mu_r
- [ ] **DROP** 城镇化率_常住人口 (100% missing)
- [ ] Identify treatment timing for each county
- [ ] Create treatment indicator (D = 1 if adopted, 0 if never)
- [ ] Create year dummies for fixed effects
- [ ] Create county dummies for fixed effects
- [ ] Subsample to non-missing mu_u and mu_r (document sample size)
- [ ] Check balance of subset sample
- [ ] Plot pre-2009 trends (2005-2008) to verify parallel trends
- [ ] Test for selection bias in Chengdu (likely similar to Chongqing)

## Files Generated

1. **ANALYSIS_SUMMARY.txt** - Comprehensive text summary of all findings
2. **README_ANALYSIS.md** - This file (markdown format)
3. **Chengdu_counties_data.xlsx** - Original data file
4. **Chongqing.xlsx** - Original data file

## Contact & Notes

This analysis was conducted to provide a comprehensive overview of the data structure, quality, and suitability for causal inference using difference-in-differences methodology.

**Key Takeaways:**
1. Both datasets have perfectly balanced panels (ideal for DiD)
2. Chongqing has excellent data quality, Chengdu needs cleaning
3. Critical selection bias exists in Chongqing (control group much richer)
4. Chengdu has longer time period but more missing data
5. Both datasets contain required treatment indicators
6. Must account for selection bias through matching/weighting in analysis

