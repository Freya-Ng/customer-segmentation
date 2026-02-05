# 🛒 Customer Segmentation Analytics Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
</p>

<p align="center">
  <b>An end-to-end machine learning project transforming raw retail transactions into actionable customer insights</b>
</p>

<p align="center">
  🔗 <a href="https://advanced-customer-segmentation.streamlit.app"><b>View Live Demo</b></a> |
  📊 <a href="#business-dashboard">Business Dashboard</a> |
  🔬 <a href="#data-science-dashboard">Data Science Dashboard</a>
</p>

---

## 🎯 Executive Summary

This project delivers a **complete customer segmentation solution** for UK online retail, combining advanced machine learning with interactive visualizations. The platform transforms **541,909 raw transactions** into **actionable customer segments**, enabling data-driven marketing strategies.

### The Challenge
> *"How can we identify distinct customer groups to personalize marketing, improve retention, and maximize customer lifetime value?"*

### The Solution
A production-ready analytics platform featuring:
- **Machine Learning Pipeline**: K-Means clustering with PCA dimensionality reduction
- **Interactive Dashboards**: Dual Streamlit dashboards for business and technical users
- **Model Interpretability**: SHAP-based explanations for cluster assignments
- **Cloud Deployment**: Accessible anywhere via Streamlit Cloud

---

## 💡 Project Motivation & Approach

### Why This Project?

Customer segmentation is one of the most practical applications of data science in business. I wanted to build something that goes beyond just running a clustering algorithm — a project that bridges the gap between technical ML work and actionable business insights.

**The questions I started with:**
- If I were a marketing manager, what would I need to know about my customers?
- Who spends the most? Who's at risk of leaving? How do I prioritize my efforts?
- How can I make ML results understandable to non-technical stakeholders?

Many segmentation projects stop at clustering — I wanted to go further with interpretability (SHAP), visualization (interactive dashboards), and real deployment (Streamlit Cloud).

### Target Audience

| Audience | What They Get |
|----------|---------------|
| **Business Stakeholders** | Insights without diving into code — KPIs, segments, recommendations |
| **Data Science Reviewers** | Full methodology transparency — validation metrics, feature analysis |
| **Learners & Practitioners** | Complete workflow reference — from raw data to deployed dashboard |

### Thinking Process & Key Decisions

| Decision | Why This Choice |
|----------|-----------------|
| **RFM Framework** | Industry-standard, intuitive for business users, captures key customer behaviors |
| **UCI Online Retail Data** | Real transactional data (not synthetic), large enough to be meaningful, publicly available |
| **Box-Cox + StandardScaler** | RFM features are heavily skewed; K-Means needs normalized, scaled features |
| **K-Means Clustering** | Simple, interpretable, works well with continuous RFM-style features |
| **Elbow + Silhouette** | Two validation methods provide confidence in K selection |
| **SHAP Analysis** | Answers "why is this customer in this cluster?" — explainability builds trust |
| **Two Dashboard Views** | Business users care about "what" and "so what"; technical users want methodology |

### Design Philosophy

- **Keep it simple** — don't over-engineer, every feature should serve a purpose
- **Every visualization answers a question** — no charts for decoration
- **Modular code** — reusable `src/` library for future projects
- **Documentation matters** — future me (and others) should understand the decisions

---

## 🏆 Key Achievements & Metrics

<table>
<tr>
<td width="50%">

### 📊 Business Impact
| Metric | Value |
|--------|-------|
| Customers Segmented | **3,920** |
| Transactions Analyzed | **354,321** |
| Customer Segments | **3-4 actionable tiers** |
| Revenue Insights | **80/20 Pareto confirmed** |

</td>
<td width="50%">

### 🔧 Technical Metrics
| Metric | Value |
|--------|-------|
| Features Engineered | **16** |
| Silhouette Score | **0.28** |
| Model Accuracy (SHAP) | **100%** |
| Interactive Charts | **17+** |

</td>
</tr>
</table>

### ✅ Project Accomplishments

- [x] Built robust ETL pipeline processing 500K+ transactions
- [x] Engineered 16 behavioral features from raw transaction data
- [x] Implemented RFM (Recency, Frequency, Monetary) analysis
- [x] Applied Box-Cox transformation for feature normalization
- [x] Optimized cluster count using Elbow + Silhouette methods
- [x] Created SHAP-based model interpretability layer
- [x] Developed 2 interactive dashboards with 17+ Plotly charts
- [x] Deployed cloud-ready Streamlit application
- [x] Wrote comprehensive documentation and insights

---

## 🌐 Live Demo

<p align="center">
  <a href="https://customer-segmentation-analytics.streamlit.app">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" width="200">
  </a>
</p>

**Access the live platform:** [https://advanced-customer-segmentation.streamlit.app](https://advanced-customer-segmentation.streamlit.app)

---

## ✨ Features Showcase

<a name="business-dashboard"></a>
### 📊 Business Analytics Dashboard

*Designed for executives, marketing teams, and business stakeholders*

| Feature | Description | Business Value |
|---------|-------------|----------------|
| **KPI Cards** | Revenue, Customers, Orders, AOV | Instant health check |
| **Revenue Trends** | Daily/Monthly with 7-day MA | Identify growth patterns |
| **Purchase Heatmap** | Day × Hour activity | Optimize campaign timing |
| **Product Analysis** | Top by Quantity & Revenue | Inventory & pricing decisions |
| **Cluster Distribution** | Revenue vs. Customer % | Segment prioritization |
| **RFM Analysis** | R, F, M distributions | Customer lifecycle stage |
| **Top Customers** | Ranked table with metrics | VIP identification |

**Automated Insights Include:**
- 📈 Trend alerts (revenue growth/decline detection)
- ⚠️ Risk warnings (customer concentration)
- 🎯 Segment recommendations (highest-value cluster)
- ⏰ Timing suggestions (peak shopping hours)
- 📊 Pareto analysis (80/20 rule validation)

<a name="data-science-dashboard"></a>
### 🔬 Data Science Dashboard

*Designed for data scientists, analysts, and technical stakeholders*

| Feature | Description | Technical Value |
|---------|-------------|-----------------|
| **Feature Analysis** | 16-feature boxplots & histograms | Distribution understanding |
| **PCA Visualization** | Variance + 2D/3D scatter | Dimensionality insights |
| **Cluster Optimization** | Elbow + Silhouette charts | Model selection rationale |
| **Radar Charts** | Cluster profile comparison | Segment characterization |
| **Feature Comparison** | Cluster means table | Quantitative differences |
| **Customer Lookup** | Individual profiling | Validation & exploration |

**Automated Insights Include:**
- 📊 PCA variance explanation (components needed)
- 🎯 Optimal K recommendation with reasoning
- 👥 Auto-generated cluster names
- 📋 Business recommendations per segment
- 🔍 Outlier detection in feature space

---

## 📈 Methodology & Technical Approach

### Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Data (541,909 transactions)                                │
│       │                                                          │
│       ├──► Remove cancelled orders (prefix 'C')                 │
│       ├──► Filter UK customers only                             │
│       ├──► Remove missing CustomerID                            │
│       ├──► Validate Quantity/Price (> 0)                        │
│       └──► Add derived columns                                  │
│                │                                                 │
│                ▼                                                 │
│  Cleaned Data (354,321 transactions, 3,920 customers)          │
│       │                                                          │
│       ├──► RFM Analysis (Recency, Frequency, Monetary)         │
│       └──► Feature Engineering (16 features)                    │
│                │                                                 │
│                ├──► Box-Cox Transformation                      │
│                └──► StandardScaler                              │
│                         │                                        │
│                         ▼                                        │
│  Modeling                                                        │
│       ├──► PCA Dimensionality Reduction                         │
│       ├──► K-Means Clustering (k=3, k=4)                        │
│       └──► SHAP Interpretation                                  │
│                │                                                 │
│                ▼                                                 │
│  Visualization (Streamlit + Plotly)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Engineering

**16 Customer-Level Features:**

| Category | Features | Purpose |
|----------|----------|---------|
| **Volume** | Sum_Quantity, Sum_TotalPrice | Total purchasing behavior |
| **Price** | Mean_UnitPrice, Mean_TotalPrice | Price sensitivity |
| **Behavior** | Count_Invoice, Count_Stock | Purchase patterns |
| **Per-Transaction** | 6 Mean_*PerInvoice metrics | Basket characteristics |
| **Per-Product** | 4 Mean_*PerStock metrics | Product engagement |

### Clustering Optimization

| Method | Finding | Conclusion |
|--------|---------|------------|
| **Elbow Method** | Inflection at k=3-4 | Diminishing returns after k=4 |
| **Silhouette Score** | Peak at k=3 (0.28) | Moderate cluster structure |
| **Business Logic** | 3-4 segments manageable | Actionable for marketing |

---

## 📊 Results & Business Insights

### Customer Segments Discovered

#### K=3 Segmentation

| Cluster | Name | % | Key Traits | Recommended Strategy |
|---------|------|---|------------|---------------------|
| **0** | Premium Customers | 39% | High unit price, quality-focused | VIP programs, exclusive access |
| **1** | Bulk Buyers | 22% | High volume, likely B2B | Volume discounts, account management |
| **2** | Diverse Shoppers | 39% | Product explorers, varied basket | Cross-sell, discovery features |

#### K=4 Segmentation

| Cluster | Name | % | Key Traits | Recommended Strategy |
|---------|------|---|------------|---------------------|
| **0** | VIP High-Value | 22% | Highest spending | White-glove service |
| **1** | Frequent Buyers | 35% | Regular purchases | Loyalty rewards |
| **2** | Budget Conscious | 15% | Price-sensitive | Value bundles, sales |
| **3** | Occasional | 29% | Infrequent | Re-engagement campaigns |

### Key Findings

1. **💰 Pareto Principle Confirmed**: Top 20% customers → ~80% revenue
2. **🏢 B2B Pattern Detected**: Peak activity Tuesday-Thursday, 10AM-3PM
3. **🎄 Seasonality**: Strong Q4 (holiday shopping surge)
4. **📦 Product Insight**: Volume leaders ≠ Revenue leaders → bundling opportunity
5. **🔑 Key Differentiators**: Sum_Quantity, Sum_TotalPrice, Mean_UnitPrice (via SHAP)

---

## 🛠 Technology Stack

### Core Technologies

| Layer | Technologies |
|-------|-------------|
| **Data Processing** | Python, Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-learn (K-Means, PCA, StandardScaler) |
| **Interpretability** | SHAP (TreeExplainer with RandomForest surrogate) |
| **Visualization** | Plotly (interactive), Matplotlib, Seaborn |
| **Dashboard** | Streamlit (multi-page app) |
| **Development** | Jupyter Notebooks |
| **Deployment** | Streamlit Cloud |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT CLOUD                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Home.py                            │   │
│  │           (Landing + Data Health Check)              │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│         ┌───────────────┴───────────────┐                  │
│         ▼                               ▼                  │
│  ┌─────────────────┐          ┌─────────────────┐         │
│  │    Business     │          │  Data Science   │         │
│  │    Analytics    │          │   Dashboard     │         │
│  └────────┬────────┘          └────────┬────────┘         │
│           │                            │                   │
│           └────────────┬───────────────┘                   │
│                        ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Shared Components                       │   │
│  │    data_loader | filters | kpi_cards | tables       │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Visualization Library                      │   │
│  │      business_charts.py  |  ds_charts.py            │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Processed Data (CSV)                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
customer-segmentation/
│
├── 📂 data/
│   ├── 📂 raw/                           # Original dataset
│   │   └── online_retail.csv             # UCI dataset (541K rows)
│   └── 📂 processed/                     # Pipeline outputs
│       ├── cleaned_uk_data.csv           # Clean transactions
│       ├── rfm_data.csv                  # RFM metrics
│       ├── customer_features.csv         # 16 features
│       ├── customer_features_scaled.csv  # Normalized
│       ├── customer_clusters_k3.csv      # K=3 results
│       └── customer_clusters_k4.csv      # K=4 results
│
├── 📂 notebooks/                         # Analysis pipeline
│   ├── 01_cleaning_and_eda.ipynb        # Cleaning + EDA + RFM
│   ├── 02_feature_engineering.ipynb     # Features + Transforms
│   └── 03_modeling.ipynb                # Clustering + SHAP
│
├── 📂 src/                              # Core library
│   ├── clustering_library.py            # ML pipeline classes
│   └── 📂 visualizations/               # Chart functions
│       ├── business_charts.py           # 8 Plotly charts
│       └── ds_charts.py                 # 9 Plotly charts
│
├── 📂 dashboards/                       # Streamlit app
│   ├── Home.py                          # Entry point
│   ├── 📂 pages/                        # Dashboard pages
│   ├── 📂 components/                   # UI components
│   └── 📂 assets/                       # CSS styling
│
├── 📂 .streamlit/                       # Streamlit config
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

---

## 🚀 Getting Started

### Quick Start (Local)

```bash
# Clone repository
git clone https://github.com/Freya-Ng/customer-segmentation.git
cd customer-segmentation

# Install dependencies
pip install -r requirements.txt

# Generate data (run notebooks in order)
jupyter notebook  # Run: 01 → 02 → 03

# Launch dashboard
cd dashboards && streamlit run Home.py
```

### Access Live Demo

🔗 **[https://advanced-customer-segmentation.streamlit.app](https://advanced-customer-segmentation.streamlit.app)**

---

## 📊 Data Source

| Attribute | Value |
|-----------|-------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) |
| **Records** | 541,909 transactions |
| **Period** | December 2010 - December 2011 |
| **Geography** | United Kingdom |
| **Industry** | Online Retail (Gifts & Homeware) |

---

## 🔮 Future Roadmap

- [ ] **Churn Prediction**: Predict at-risk customers
- [ ] **CLV Forecasting**: Customer lifetime value models
- [ ] **Real-time Pipeline**: Live data integration
- [ ] **API Layer**: REST endpoints for CRM integration
- [ ] **A/B Testing Module**: Campaign effectiveness tracking
- [ ] **Recommendation Engine**: Personalized product suggestions

---

## 👤 About the Author

**Nguyen An Phuong Linh**

Data Scientist passionate about turning data into actionable insights.

- 🔗 GitHub: [@Freya-Ng](https://github.com/Freya-Ng)
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/)

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  <b>⭐ Star this repository if you found it useful! ⭐</b>
</p>

<p align="center">
  <i>Built with Python, Streamlit, and lots of ☕</i>
</p>
