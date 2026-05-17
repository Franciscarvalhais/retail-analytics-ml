# Retail Analytics — Machine Learning Models

## Overview
Application of three Machine Learning models to the Online Retail Dataset (UCI/Kaggle), containing 541,910 transactions from a UK-based online retailer. Each model addresses a concrete business question: sales forecasting, customer retention prediction, and customer segmentation.

## Dataset
- **Source:** Online Retail Dataset (UCI Machine Learning Repository / Kaggle)
- **Size:** 541,910 transactions → 401,565 after cleaning
- **Period:** December 2010 – December 2011
- **Scope:** 3,877 products · 4,338 customers · 38 countries

## Data Cleaning
- Removed 135,080 rows with missing Customer ID (25% of dataset)
- Investigated and removed 5,225 confirmed duplicate entries
- Created new variables: TotalPrice, transaction type (Purchase/Return)
- Final return rate: 2.21% — consistent with online retail benchmarks

## Models

### Model 1 — Regression: Monthly Sales Forecasting
- **Goal:** Predict monthly units sold per product to support stock management
- **Method:** Random Forest Regressor with log transformation
- **Features:** 8 engineered features including lag variables and seasonality indicators
- **Result:** R² = 0.86 (log scale) · MAE = 70 units
- **Key finding:** Number of transactions (66.7%) is the strongest predictor — calendar seasonality has minimal direct impact

### Model 2 — Classification: Recurring vs One-time Customers
- **Goal:** Identify potential recurring customers from their first purchase
- **Method:** Random Forest Classifier with threshold analysis
- **Features:** First-purchase attributes only (simulating real-world scenario)
- **Result:** Accuracy = 71.4% · AUC = 0.72
- **Key finding:** Recurring customers prefer lower-priced items (£3.90 vs £5.76 average)

### Model 3 — Clustering: RFM Customer Segmentation
- **Goal:** Segment customers for personalised marketing strategies
- **Method:** K-Means with RFM methodology (Recency, Frequency, Monetary)
- **Result:** 4 segments identified (Silhouette Score = 0.34)
- **Key finding:** 16% of customers (VIP) generate 65% of total revenue — Pareto principle confirmed

## Customer Segments
| Segment | Customers | Recency | Frequency | Monetary |
|---------|-----------|---------|-----------|----------|
| VIP | 713 | 12 days | 14 | £8,088 |
| At Risk | 1,622 | 182 days | 1.3 | £341 |
| Loyal (low) | 837 | 18 days | 2.2 | £557 |
| Loyal (mid) | 1,166 | 72 days | 4.0 | £1,801 |

## Technologies
Python · scikit-learn · pandas · numpy · matplotlib · seaborn

## Context
Developed for the Business Intelligence course, MSc in Data Engineering and Science at UTAD (University of Trás-os-Montes e Alto Douro), May 2026.
