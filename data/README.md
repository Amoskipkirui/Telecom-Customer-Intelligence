# Telecom Customer Intelligence — Synthetic Raw Dataset

This dataset is entirely synthetic and is not real Ooredoo or any other telecom operator customer data.

## Files
- `customers_raw.csv`: customer profile, plan, contract, tenure, city and service details.
- `usage_raw.csv`: monthly usage and network experience.
- `billing_raw.csv`: billing, total charges and payment delays.
- `support_raw.csv`: complaints and support contacts.
- `churn_target_raw.csv`: churn outcome (`Yes`/`No`) for supervised learning.
- `monthly_customer_revenue_raw.csv`: monthly customer revenue across 12 months (Oct 2025–Sep 2026).

## Intended workflow
1. Import each CSV into its own raw MySQL table. Keep the raw versions unchanged.
2. Use SQL to profile quality: nulls, duplicates, inconsistent categories, invalid values and missing customer IDs across tables.
3. Create cleaned/staging tables or views using SQL.
4. Build a `customer_analytics` view by joining the cleaned tables on `customer_id`.
5. Read `customer_analytics` into Pandas for EDA, feature engineering and model development.
6. Use `monthly_customer_revenue_raw.csv` for time-based revenue analysis if desired.

## Deliberate data-quality issues
The raw files intentionally include some missing values, duplicate rows, inconsistent city/plan labels, invalid negative values in selected measures, and missing records in related tables. Investigate before cleaning; do not blindly delete all nulls or duplicates.

## Important
- Customer IDs are synthetic.
- Values and churn labels are simulated for learning and portfolio development.
- Do not represent this as actual operator data.
