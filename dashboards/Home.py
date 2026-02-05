"""
Customer Segmentation Dashboard - Home Page

Main entry point for the Streamlit multi-page application.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.components.data_loader import check_data_files, load_transactions, load_rfm, load_clusters
from dashboards.components.kpi_cards import render_kpi_row, calculate_business_kpis


def load_css():
    """Load custom CSS styling."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    # Page configuration
    st.set_page_config(
        page_title="Customer Segmentation Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    load_css()

    # Header
    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("### UK Online Retail Analysis (Dec 2010 - Dec 2011)")

    st.markdown("---")

    # Data health check
    st.subheader("Data Status")
    file_status = check_data_files()

    if not all(file_status.values()):
        st.error("⚠️ Missing data files. Please run the data processing notebooks first.")
        st.markdown("**Required files:**")
        for file, exists in file_status.items():
            icon = "✅" if exists else "❌"
            st.write(f"{icon} `data/processed/{file}`")

        st.markdown("""
        **To generate the required data files, run the notebooks in order:**
        1. `notebooks/01_cleaning_and_eda.ipynb` - Generates cleaned_uk_data.csv and rfm_data.csv
        2. `notebooks/02_feature_engineering.ipynb` - Generates customer_features.csv and customer_features_scaled.csv
        3. `notebooks/03_modeling.ipynb` - Generates customer_clusters_k3.csv and customer_clusters_k4.csv
        """)
        return

    # All files present
    st.success("✅ All data files found. Dashboards are ready to use.")

    # Show file status in expandable section
    with st.expander("View Data Files"):
        for file, exists in file_status.items():
            st.write(f"✅ `data/processed/{file}`")

    st.markdown("---")

    # Overview KPIs
    st.subheader("Dataset Overview")

    try:
        transactions = load_transactions()
        rfm = load_rfm()
        clusters = load_clusters(k=3)

        kpis = calculate_business_kpis(transactions)

        render_kpi_row([
            {"label": "Total Revenue", "value": kpis["total_revenue"], "format": "currency"},
            {"label": "Total Customers", "value": kpis["total_customers"], "format": "number"},
            {"label": "Total Orders", "value": kpis["total_orders"], "format": "number"},
            {"label": "Avg Order Value", "value": kpis["avg_order_value"], "format": "currency"},
        ])

        st.markdown("---")

        # Cluster summary
        st.subheader("Cluster Summary (k=3)")
        cluster_counts = clusters.groupby("Cluster").size().reset_index(name="Customers")
        cluster_counts["Percentage"] = (cluster_counts["Customers"] / cluster_counts["Customers"].sum() * 100).round(1)

        cols = st.columns(len(cluster_counts))
        for col, (_, row) in zip(cols, cluster_counts.iterrows()):
            with col:
                st.metric(
                    label=f"Cluster {int(row['Cluster'])}",
                    value=f"{int(row['Customers']):,}",
                    delta=f"{row['Percentage']}%",
                    delta_color="off"
                )

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

    st.markdown("---")

    # Navigation cards
    st.subheader("Explore Dashboards")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        ### 📊 Business Analytics

        **For Business Stakeholders**

        - Revenue trends and forecasts
        - Customer behavior analysis
        - Product performance insights
        - RFM analysis
        - Top customers ranking

        👉 Use the sidebar to navigate to **Business Analytics**
        """)

    with col2:
        st.info("""
        ### 🔬 Data Science

        **For Technical Users**

        - Feature engineering analysis
        - PCA visualization
        - Cluster optimization (Elbow, Silhouette)
        - Cluster profiles (Radar charts)
        - Customer lookup tool

        👉 Use the sidebar to navigate to **Data Science**
        """)

    st.markdown("---")

    # About section
    with st.expander("About This Project"):
        st.markdown("""
        **Customer Segmentation Dashboard** is an interactive analytics tool for understanding
        customer behavior in UK online retail.

        **Data Source:** UCI Machine Learning Repository - Online Retail Dataset

        **Methodology:**
        1. Data cleaning and preprocessing
        2. RFM (Recency, Frequency, Monetary) analysis
        3. Feature engineering (16 customer-level features)
        4. K-Means clustering with PCA
        5. SHAP-based interpretability

        **Tech Stack:**
        - Python, Pandas, Scikit-learn
        - Streamlit, Plotly
        - SHAP for model interpretability
        """)


if __name__ == "__main__":
    main()
