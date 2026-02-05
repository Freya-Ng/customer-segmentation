# Customer Segmentation Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-segmentation-analytics.streamlit.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

An end-to-end machine learning pipeline for customer segmentation using RFM analysis, K-Means clustering, and SHAP explainability on retail transaction data.

🔗 **Live Dashboard:** [https://customer-segmentation-analytics.streamlit.app](https://customer-segmentation-analytics.streamlit.app)

## Overview

This project analyzes **541,909 retail transactions** to segment **3,920 UK customers** into distinct behavioral groups, enabling targeted marketing strategies and personalized customer experiences.

## Techniques & Methods

| Stage | Technique |
|-------|-----------|
| Data Cleaning | Outlier removal, missing value handling, cancelled invoice filtering |
| Feature Engineering | RFM metrics, 16 behavioral features, temporal patterns (DayOfWeek, HourOfDay) |
| Transformation | Box-Cox normalization, StandardScaler standardization |
| Dimensionality Reduction | Principal Component Analysis (PCA) |
| Clustering | K-Means (k=3, k=4), Elbow method, Silhouette analysis |
| Explainability | RandomForest surrogate model, SHAP TreeExplainer |

## Key Results

### Data Pipeline
- **Raw data:** 541,909 transactions
- **Cleaned data:** 354,321 transactions (34.6% removed)
- **Customer base:** 3,920 unique UK customers
- **Features:** 16 engineered customer-level metrics

### Customer Segments (K=3)

| Segment | Size | Characteristics |
|---------|------|-----------------|
| **Premium Customers** | 39.3% | High unit price (~£6.35 avg), quality-focused, fewer transactions |
| **Bulk Buyers** | 22.2% | High volume (~2,627 items avg), frequent purchases, likely B2B |
| **Diverse Shoppers** | 38.5% | High product diversity (~95 unique products), exploratory behavior |

### Model Performance
- Surrogate model accuracy: **100%** (perfect K-Means replication)
- SHAP analysis reveals distinct feature importance per cluster

## Key Findings

1. **Revenue Distribution:** Bulk buyers drive the majority of revenue despite being the smallest segment (22.2%)
2. **Price Sensitivity:** Premium customers show low price sensitivity with ~2.4x higher average unit prices
3. **Behavioral Patterns:** Clear separation between volume-driven (B2B) and diversity-driven (retail) customers
4. **Optimal Segmentation:** K=4 provides finer granularity by splitting buyers into small/large accounts
5. **Feature Importance:** SHAP reveals Sum_Quantity, Sum_TotalPrice, and Mean_UnitPrice as top differentiators

## Interactive Dashboards

This project includes **two interactive Streamlit dashboards** for exploring the segmentation results:

### 📊 Business Analytics Dashboard
For business stakeholders and marketing teams:
- KPI cards (Revenue, Customers, Orders, AOV)
- Daily/Monthly revenue trends with moving averages
- Purchase heatmap (Day × Hour patterns)
- Top products analysis (Quantity & Revenue)
- Customer distribution and Pareto analysis
- RFM distributions with actionable insights

### 🔬 Data Science Dashboard
For analysts and data scientists:
- Feature distributions (Boxplots & Histograms)
- PCA visualization (2D/3D scatter plots)
- Cluster optimization (Elbow & Silhouette)
- Radar charts for cluster profiles
- Customer lookup tool
- Feature comparison tables

**Run the dashboard locally:**
```bash
cd dashboards
streamlit run Home.py
```

## Project Structure

```
customer_segmentation/
├── data/
│   ├── raw/                           # Original dataset
│   │   └── online_retail.csv
│   └── processed/                     # Transformed outputs
│       ├── cleaned_uk_data.csv
│       ├── rfm_data.csv
│       ├── customer_features.csv
│       ├── customer_features_transformed.csv
│       ├── customer_features_scaled.csv
│       ├── customer_clusters_k3.csv
│       └── customer_clusters_k4.csv
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb      # Data cleaning & exploration
│   ├── 02_feature_engineering.ipynb   # Feature creation & transformation
│   └── 03_modeling.ipynb              # Clustering & SHAP interpretation
├── src/
│   ├── __init__.py
│   ├── clustering_library.py          # Core ML pipeline classes
│   └── visualizations/                # Plotly chart functions
│       ├── __init__.py
│       ├── business_charts.py         # Business dashboard charts
│       └── ds_charts.py               # Data science dashboard charts
├── dashboards/                        # Streamlit application
│   ├── Home.py                        # Main entry point
│   ├── pages/
│   │   ├── 1_Business_Analytics.py    # Business dashboard
│   │   └── 2_Data_Science.py          # Data science dashboard
│   ├── components/                    # Reusable UI components
│   │   ├── data_loader.py
│   │   ├── filters.py
│   │   ├── kpi_cards.py
│   │   └── tables.py
│   └── assets/
│       └── style.css
├── .streamlit/
│   └── config.toml                    # Streamlit configuration
├── requirements.txt                   # Python dependencies
└── README.md
```

## Tech Stack

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn (KMeans, PCA, RandomForestClassifier, StandardScaler)
- **Visualization:** matplotlib, seaborn, plotly
- **Dashboard:** streamlit
- **Explainability:** shap (TreeExplainer)
- **Statistical:** scipy (Box-Cox transformation)

## Quick Start

```bash
# Clone and navigate to project
cd customer_segmentation

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order to generate processed data
# 1. notebooks/01_cleaning_and_eda.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_modeling.ipynb

# Launch the interactive dashboard
cd dashboards
streamlit run Home.py
```

**Or access the live demo:** [https://customer-segmentation-analytics.streamlit.app](https://customer-segmentation-analytics.streamlit.app)

## Dataset

**Source:** UCI Online Retail Dataset
**Period:** December 2010 - December 2011 (12 months)
**Scope:** UK-based online gift retailer transactions

### Data Fields
| Field | Description |
|-------|-------------|
| InvoiceNo | 6-digit invoice identifier |
| StockCode | Product code |
| Description | Product name |
| Quantity | Items purchased |
| InvoiceDate | Transaction timestamp |
| UnitPrice | Price in GBP |
| CustomerID | Customer identifier |
| Country | Customer country |
