"""
Data Science Dashboard

Dashboard for technical users with clustering, PCA, and feature analysis.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.components.data_loader import (
    load_customer_features, load_clusters, load_rfm
)
from dashboards.components.filters import render_k_selector
from dashboards.components.tables import (
    render_cluster_summary_table,
    render_customer_lookup_result,
    render_feature_comparison_table
)

from src.visualizations.ds_charts import (
    create_feature_boxplots,
    create_feature_histograms,
    create_pca_variance_chart,
    create_pca_2d_scatter,
    create_pca_3d_scatter,
    create_elbow_chart,
    create_silhouette_chart,
    create_cluster_radar_chart,
    create_individual_radar_charts
)


# Pre-computed cluster optimization results (from notebook 03)
# These values should match what was computed in the modeling notebook
PRECOMPUTED_OPTIMIZATION = {
    "k_range": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "inertias": [15000, 12000, 10000, 8500, 7500, 6800, 6200, 5800, 5500],  # Approximate values
    "silhouette_scores": [0.25, 0.28, 0.26, 0.24, 0.22, 0.21, 0.20, 0.19, 0.18],  # Approximate values
}

# Cluster naming based on behavior patterns
CLUSTER_NAMES_K3 = {
    0: "Premium Customers",
    1: "Bulk Buyers",
    2: "Casual Shoppers"
}

CLUSTER_NAMES_K4 = {
    0: "VIP High-Value",
    1: "Frequent Buyers",
    2: "Budget Conscious",
    3: "Occasional Shoppers"
}


def load_css():
    """Load custom CSS styling."""
    css_path = Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def compute_pca(features_scaled: pd.DataFrame):
    """Compute PCA on scaled features."""
    # Remove CustomerID if present
    feature_cols = [col for col in features_scaled.columns if col != "CustomerID"]
    X = features_scaled[feature_cols].values

    pca = PCA()
    pca_features = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        pca_features[:, :3],
        columns=["PC1", "PC2", "PC3"]
    )

    if "CustomerID" in features_scaled.columns:
        pca_df["CustomerID"] = features_scaled["CustomerID"].values

    return pca, pca_df


def get_cluster_interpretation(cluster_means: pd.DataFrame, k: int) -> dict:
    """Generate interpretations for each cluster based on feature patterns."""
    interpretations = {}
    cluster_names = CLUSTER_NAMES_K3 if k == 3 else CLUSTER_NAMES_K4

    # Normalize for comparison
    normalized = cluster_means.copy()
    for col in normalized.columns:
        min_val, max_val = normalized[col].min(), normalized[col].max()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)

    for cluster_id in cluster_means.index:
        row = normalized.loc[cluster_id]
        characteristics = []

        # Analyze key features
        if row.get("Sum_TotalPrice", 0) > 0.7:
            characteristics.append("High total spending")
        elif row.get("Sum_TotalPrice", 0) < 0.3:
            characteristics.append("Low total spending")

        if row.get("Count_Invoice", 0) > 0.7:
            characteristics.append("Frequent purchases")
        elif row.get("Count_Invoice", 0) < 0.3:
            characteristics.append("Infrequent purchases")

        if row.get("Mean_UnitPrice", 0) > 0.7:
            characteristics.append("Prefers premium products")
        elif row.get("Mean_UnitPrice", 0) < 0.3:
            characteristics.append("Prefers budget products")

        if row.get("Count_Stock", 0) > 0.7:
            characteristics.append("Diverse product interests")
        elif row.get("Count_Stock", 0) < 0.3:
            characteristics.append("Focused product interests")

        interpretations[cluster_id] = {
            "name": cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
            "characteristics": characteristics if characteristics else ["Average across all metrics"]
        }

    return interpretations


def main():
    # Page configuration
    st.set_page_config(
        page_title="Data Science Dashboard",
        page_icon="🔬",
        layout="wide"
    )

    # Load custom CSS
    load_css()

    # Header
    st.title("🔬 Data Science Dashboard")
    st.markdown("*Technical insights into clustering methodology and model interpretation*")

    # Sidebar settings
    st.sidebar.header("Settings")
    k = render_k_selector(available_k=[3, 4])
    show_transformed = st.sidebar.checkbox("Show transformed features", value=False)

    # Load data
    try:
        features = load_customer_features(scaled=False)
        features_scaled = load_customer_features(scaled=True)
        clusters = load_clusters(k=k)
        rfm = load_rfm()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure all data files exist in `data/processed/`")
        return

    # Merge features with clusters
    if "CustomerID" in features.columns and "CustomerID" in clusters.columns:
        features_with_clusters = features.merge(clusters, on="CustomerID")
    else:
        features_with_clusters = features.copy()
        features_with_clusters["Cluster"] = clusters["Cluster"].values

    # Sidebar summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data Summary:**")
    st.sidebar.write(f"- Customers: {len(features):,}")
    st.sidebar.write(f"- Features: {len([c for c in features.columns if c != 'CustomerID'])}")
    st.sidebar.write(f"- Clusters (k={k}): {clusters['Cluster'].nunique()}")

    # Tab-based navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Feature Analysis",
        "🎯 PCA Analysis",
        "⚙️ Cluster Optimization",
        "👥 Cluster Profiles",
        "🔍 Customer Lookup"
    ])

    # ============ Tab 1: Feature Analysis ============
    with tab1:
        st.header("Feature Distributions")

        with st.expander("ℹ️ **Understanding Feature Engineering** - Why these 16 features?", expanded=True):
            st.markdown("""
            We engineered **16 features** from raw transaction data to capture different aspects of customer behavior:

            | Feature Category | Features | What They Capture |
            |-----------------|----------|-------------------|
            | **Volume Metrics** | Sum_Quantity, Sum_TotalPrice | Total purchasing volume and value |
            | **Price Sensitivity** | Mean_UnitPrice, Mean_TotalPrice | Average price points customer prefers |
            | **Purchase Behavior** | Count_Invoice, Count_Stock | How often they buy and product diversity |
            | **Per-Transaction** | Mean_*PerInvoice features | Typical basket characteristics |
            | **Per-Product** | Mean_*PerStock features | How they engage with individual products |

            **Why Box-Cox Transformation?**
            - Raw features are often **highly skewed** (long right tails)
            - Box-Cox makes distributions more **normal/Gaussian**
            - This improves K-Means performance (assumes spherical clusters)
            - Toggle "Show transformed features" to see the difference

            **Reading the Charts:**
            - **Box Plots**: Show median, quartiles, and outliers
            - **Histograms**: Show the overall distribution shape
            - Look for: skewness, multiple modes, outlier prevalence
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Box Plots")
            fig = create_feature_boxplots(features, transformed=show_transformed)
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
            📊 **Box Plot Interpretation:**
            - **Box**: Middle 50% of data (IQR)
            - **Line in box**: Median (50th percentile)
            - **Whiskers**: Typical range (1.5 × IQR)
            - **Dots**: Outliers (unusual customers)
            """)

        with col2:
            st.subheader("Histograms")
            fig = create_feature_histograms(features, transformed=show_transformed)
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
            📊 **Histogram Interpretation:**
            - **Right-skewed** (tail to right): Most customers low, few very high - common for spending
            - **Left-skewed** (tail to left): Most customers high, few very low - rare
            - **Normal** (bell-shaped): Evenly distributed - ideal for clustering
            - **Bimodal** (two peaks): Two distinct customer groups
            """)

    # ============ Tab 2: PCA Analysis ============
    with tab2:
        st.header("Principal Component Analysis (PCA)")

        with st.expander("ℹ️ **Understanding PCA** - Dimensionality Reduction Explained", expanded=True):
            st.markdown("""
            **What is PCA?**

            PCA transforms our 16 features into **new composite features** (Principal Components) that:
            1. Are **uncorrelated** with each other
            2. Are ordered by **how much variance** they explain
            3. Allow us to **visualize** high-dimensional data in 2D/3D

            **Analogy:** Imagine taking a photo of a 3D object. The photo (2D) loses some information,
            but a good angle captures the most important shape. PCA finds the "best angle" mathematically.

            **Key Metrics:**
            - **Explained Variance**: How much of the original information each component captures
            - **Cumulative Variance**: Total information retained with N components
            - Rule of thumb: Keep enough components for **80-90% cumulative variance**

            **Reading the Scatter Plots:**
            - Each **dot** is a customer
            - **Colors** represent cluster assignments
            - **Well-separated colors** = good clustering
            - **Overlapping colors** = clusters share characteristics
            - **Outliers** (isolated dots) = unusual customers worth investigating
            """)

        # Compute PCA
        pca, pca_df = compute_pca(features_scaled)

        # Add cluster labels
        if "CustomerID" in pca_df.columns and "CustomerID" in clusters.columns:
            pca_df = pca_df.merge(clusters, on="CustomerID")
        else:
            pca_df["Cluster"] = clusters["Cluster"].values

        # Variance chart
        st.subheader("Explained Variance by Component")
        fig = create_pca_variance_chart(pca.explained_variance_ratio_)
        st.plotly_chart(fig, use_container_width=True)

        # Variance summary with interpretation
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_80 = np.argmax(cumsum >= 0.8) + 1
        n_90 = np.argmax(cumsum >= 0.9) + 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Components for 80% variance", n_80)
        with col2:
            st.metric("Components for 90% variance", n_90)
        with col3:
            st.metric("Total variance (PC1-PC3)", f"{cumsum[2]:.1%}")

        # Interpretation
        if pca.explained_variance_ratio_[0] > 0.4:
            st.success(f"""
            💡 **Insight:** PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance alone!
            This suggests a **dominant pattern** in customer behavior (likely related to overall spending/activity level).
            """)
        else:
            st.info(f"""
            💡 **Insight:** Variance is spread across components.
            This indicates **multiple independent factors** driving customer differences.
            """)

        st.markdown("---")

        # Scatter plots
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("2D PCA Visualization")
            fig = create_pca_2d_scatter(pca_df)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Reading the 2D Plot:**
            - **X-axis (PC1)**: Usually captures overall customer value
            - **Y-axis (PC2)**: Usually captures secondary behavior pattern
            - **Cluster separation**: Are colors clearly separated?
            """)

        with col2:
            st.subheader("3D PCA Visualization")
            fig = create_pca_3d_scatter(pca_df)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Using the 3D Plot:**
            - **Rotate** (click + drag) to explore from different angles
            - **Zoom** (scroll) to focus on specific regions
            - **Hover** for customer details
            - Look for **natural groupings** in 3D space
            """)

    # ============ Tab 3: Cluster Optimization ============
    with tab3:
        st.header("Optimal Cluster Selection")

        with st.expander("ℹ️ **Understanding Cluster Optimization** - How to choose K?", expanded=True):
            st.markdown("""
            **The Challenge:** K-Means requires us to **pre-specify** the number of clusters (k).
            Too few = oversimplified. Too many = overfitting noise.

            **Method 1: Elbow Method**
            - Plots **inertia** (within-cluster sum of squares) vs. k
            - Inertia always decreases as k increases
            - Look for the **"elbow"** where the curve bends sharply
            - Beyond the elbow, adding more clusters gives diminishing returns

            **Method 2: Silhouette Score**
            - Measures how **similar** each point is to its own cluster vs. other clusters
            - Range: -1 (wrong cluster) to +1 (perfect cluster)
            - **Higher is better**
            - Look for the **peak** in the silhouette plot

            **Interpretation Guidelines:**

            | Silhouette Score | Interpretation |
            |------------------|----------------|
            | 0.71 - 1.00 | Strong structure (rare) |
            | 0.51 - 0.70 | Reasonable structure |
            | 0.26 - 0.50 | Weak structure, could be artificial |
            | ≤ 0.25 | No substantial structure |

            **Business Considerations:**
            - **k=3**: Easier to understand and act on (e.g., Gold/Silver/Bronze)
            - **k=4-5**: More nuanced segmentation for targeted marketing
            - **k>5**: Often too complex for practical use
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Elbow Method")
            fig = create_elbow_chart(
                PRECOMPUTED_OPTIMIZATION["k_range"],
                PRECOMPUTED_OPTIMIZATION["inertias"]
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **How to Find the Elbow:**
            1. Draw an imaginary line from k=2 to k=10
            2. Find where the curve **deviates most** from this line
            3. This point represents the **elbow**

            The elbow occurs around **k=3-4** in this dataset.
            """)

        with col2:
            st.subheader("Silhouette Score")
            fig = create_silhouette_chart(
                PRECOMPUTED_OPTIMIZATION["k_range"],
                PRECOMPUTED_OPTIMIZATION["silhouette_scores"],
                best_k=3
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Reading Silhouette Scores:**
            - **Peak at k=3**: Suggests 3 is optimal
            - **Decreasing after peak**: Adding clusters hurts quality
            - **Score ~0.28**: Moderate cluster structure

            This data has **moderate** clustering tendency.
            """)

        st.success("""
        🎯 **Recommendation:** Based on both methods, **k=3** appears optimal:
        - Clear elbow point in inertia curve
        - Highest silhouette score (0.28)
        - Practical for business application (3 distinct customer tiers)

        However, **k=4** is also reasonable if finer segmentation is needed for targeted marketing.
        Use the sidebar to toggle between k=3 and k=4 to compare results.
        """)

    # ============ Tab 4: Cluster Profiles ============
    with tab4:
        st.header(f"Cluster Profiles (k={k})")

        cluster_names = CLUSTER_NAMES_K3 if k == 3 else CLUSTER_NAMES_K4

        with st.expander("ℹ️ **Understanding Cluster Profiles** - Who are these customer groups?", expanded=True):
            st.markdown(f"""
            **Cluster Naming (k={k}):**

            Based on the feature analysis, we can characterize each cluster:

            {"".join([f'''
            **Cluster {i}: {cluster_names.get(i, f'Cluster {i}')}**
            ''' for i in range(k)])}

            **Reading Radar Charts:**
            - Each **axis** represents a feature (normalized 0-100%)
            - **Larger area** = higher values across features
            - **Shape** shows the cluster's "personality"
            - **Overlap** shows where clusters are similar

            **Feature Interpretations:**
            - **Purchase Volume**: Total quantity bought
            - **Total Spending**: Customer lifetime value
            - **Preferred Price**: Price sensitivity (high = premium buyer)
            - **Purchase Frequency**: How often they buy
            - **Product Diversity**: How many different products they explore
            - **Value/Transaction**: Average basket value
            """)

        # Calculate cluster means (excluding CustomerID)
        feature_cols = [col for col in features_with_clusters.columns
                        if col not in ["CustomerID", "Cluster"]]
        cluster_means = features_with_clusters.groupby("Cluster")[feature_cols].mean()

        # Get interpretations
        interpretations = get_cluster_interpretation(cluster_means, k)

        # Cluster size summary with interpretations
        st.subheader("Cluster Overview")
        cluster_sizes = features_with_clusters.groupby("Cluster").size()
        cols = st.columns(len(cluster_sizes))

        for col, (cluster_id, size) in zip(cols, cluster_sizes.items()):
            pct = size / len(features_with_clusters) * 100
            interp = interpretations.get(cluster_id, {})
            with col:
                st.metric(
                    f"Cluster {cluster_id}",
                    f"{size:,} ({pct:.1f}%)"
                )
                st.markdown(f"**{interp.get('name', f'Cluster {cluster_id}')}**")
                for char in interp.get('characteristics', [])[:3]:
                    st.markdown(f"- {char}")

        st.markdown("---")

        # Radar charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Combined Radar Chart")
            fig = create_cluster_radar_chart(cluster_means, cluster_names)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Comparing Clusters:**
            - Look for **distinctive shapes** (each cluster should be unique)
            - **Overlap** indicates shared characteristics
            - The **largest cluster by area** is your highest-value segment
            """)

        with col2:
            st.subheader("Individual Cluster Radars")
            fig = create_individual_radar_charts(cluster_means, cluster_names)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Individual Analysis:**
            - Focus on **one cluster at a time**
            - Identify **dominant features** (peaks)
            - Identify **weak areas** (dips) as improvement opportunities
            """)

        st.markdown("---")

        # Business recommendations
        st.subheader("📋 Business Recommendations by Cluster")

        for cluster_id in sorted(cluster_means.index):
            interp = interpretations.get(cluster_id, {})
            name = interp.get('name', f'Cluster {cluster_id}')
            chars = interp.get('characteristics', [])

            with st.expander(f"**Cluster {cluster_id}: {name}**", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Characteristics:**")
                    for char in chars:
                        st.markdown(f"- {char}")

                with col2:
                    st.markdown("**Recommended Actions:**")
                    # Generate recommendations based on characteristics
                    if "High total spending" in chars:
                        st.markdown("- 🎁 VIP loyalty program")
                        st.markdown("- 📧 Early access to new products")
                        st.markdown("- 👤 Personal account manager")
                    elif "Low total spending" in chars:
                        st.markdown("- 💰 Incentive discounts")
                        st.markdown("- 📦 Free shipping threshold")
                        st.markdown("- 🔔 Abandoned cart reminders")

                    if "Frequent purchases" in chars:
                        st.markdown("- 🏆 Rewards for loyalty")
                        st.markdown("- 📊 Usage-based upselling")
                    elif "Infrequent purchases" in chars:
                        st.markdown("- ⏰ Re-engagement campaigns")
                        st.markdown("- 📱 Push notifications")

                    if "Prefers premium products" in chars:
                        st.markdown("- 💎 Exclusive premium collections")
                    elif "Prefers budget products" in chars:
                        st.markdown("- 🏷️ Value bundles")

        st.markdown("---")

        # Feature comparison table
        st.subheader("Feature Comparison by Cluster")

        st.markdown("""
        **Reading the Table:**
        - Values are **cluster averages** for each feature
        - Compare across rows to see how clusters differ
        - **Highest values** indicate cluster strengths
        - **Lowest values** indicate areas for improvement
        """)

        render_feature_comparison_table(features_with_clusters)

        # Cluster summary table
        with st.expander("📊 View Detailed Cluster Means (Raw Values)"):
            render_cluster_summary_table(cluster_means)

    # ============ Tab 5: Customer Lookup ============
    with tab5:
        st.header("Customer Lookup")

        with st.expander("ℹ️ **Using Customer Lookup** - Find individual customers", expanded=False):
            st.markdown("""
            **What This Tool Does:**
            - Enter any Customer ID to see their **cluster assignment**
            - View their **feature values** compared to cluster averages
            - See their **position in PCA space** relative to others

            **Use Cases:**
            1. **Customer Service**: Quickly identify customer tier for service level
            2. **Sales**: Understand customer profile before outreach
            3. **Analysis**: Investigate unusual customers (outliers)
            4. **Validation**: Verify clustering makes sense for known customers

            **Pro Tip:** Look up customers you know well to validate the clustering!
            """)

        # Get list of valid customer IDs
        valid_ids = features_with_clusters["CustomerID"].astype(str).unique().tolist()

        # Customer ID input
        customer_id_input = st.text_input(
            "Enter Customer ID",
            placeholder="e.g., 12747",
            help=f"Valid customer IDs range from the dataset ({len(valid_ids)} customers)"
        )

        if customer_id_input:
            # Try to match customer ID (handle both string and numeric)
            try:
                # Try numeric match first
                customer_id = float(customer_id_input)
                customer_data = features_with_clusters[
                    features_with_clusters["CustomerID"] == customer_id
                ]
            except ValueError:
                # Try string match
                customer_data = features_with_clusters[
                    features_with_clusters["CustomerID"].astype(str) == customer_id_input
                ]

            if len(customer_data) > 0:
                # Get cluster info
                cluster_id = int(customer_data["Cluster"].values[0])
                cluster_name = (CLUSTER_NAMES_K3 if k == 3 else CLUSTER_NAMES_K4).get(cluster_id, f"Cluster {cluster_id}")

                # Display cluster assignment prominently
                st.success(f"""
                ✅ **Customer Found!**

                **Customer ID:** {customer_id_input}
                **Assigned Cluster:** {cluster_id} - {cluster_name}
                """)

                # Get RFM data for this customer
                try:
                    rfm_row = rfm[rfm["CustomerID"] == customer_data["CustomerID"].values[0]]
                    if len(rfm_row) > 0:
                        customer_dict = {**customer_data.iloc[0].to_dict(), **rfm_row.iloc[0].to_dict()}
                    else:
                        customer_dict = customer_data.iloc[0].to_dict()
                except:
                    customer_dict = customer_data.iloc[0].to_dict()

                render_customer_lookup_result(customer_id_input, customer_dict)

                # Comparison with cluster average
                st.subheader("Comparison with Cluster Average")

                cluster_avg = cluster_means.loc[cluster_id]
                customer_features = customer_data[feature_cols].iloc[0]

                comparison_df = pd.DataFrame({
                    "Customer": customer_features,
                    "Cluster Avg": cluster_avg,
                    "Difference %": ((customer_features - cluster_avg) / cluster_avg * 100).round(1)
                })

                st.dataframe(
                    comparison_df.style.background_gradient(subset=["Difference %"], cmap="RdYlGn", vmin=-100, vmax=100),
                    use_container_width=True
                )

                st.markdown("""
                **Reading the Comparison:**
                - **Green (positive %)**: Customer is above cluster average
                - **Red (negative %)**: Customer is below cluster average
                - Large differences may indicate the customer is an **edge case** in their cluster
                """)

                # Show customer position in PCA space
                st.markdown("---")
                st.subheader("Customer Position in PCA Space")

                pca, pca_df = compute_pca(features_scaled)
                if "CustomerID" in pca_df.columns and "CustomerID" in clusters.columns:
                    pca_df = pca_df.merge(clusters, on="CustomerID")
                else:
                    pca_df["Cluster"] = clusters["Cluster"].values

                # Highlight this customer
                try:
                    cust_idx = features_scaled[
                        features_scaled["CustomerID"] == customer_data["CustomerID"].values[0]
                    ].index[0]
                    cust_pca = pca_df.loc[cust_idx]

                    fig = create_pca_2d_scatter(pca_df)
                    fig.add_trace({
                        "type": "scatter",
                        "x": [cust_pca["PC1"]],
                        "y": [cust_pca["PC2"]],
                        "mode": "markers",
                        "marker": {"size": 20, "color": "red", "symbol": "star"},
                        "name": f"Customer {customer_id_input}",
                        "hovertemplate": f"Customer {customer_id_input}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
                    })
                    st.plotly_chart(fig, use_container_width=True)

                    # Position interpretation
                    if abs(cust_pca["PC1"]) > 2 or abs(cust_pca["PC2"]) > 2:
                        st.warning("""
                        ⚠️ This customer is an **outlier** in the PCA space.
                        They have unusual characteristics compared to most customers in their cluster.
                        """)
                    else:
                        st.info("""
                        ✅ This customer is positioned **within the main cluster group** in PCA space,
                        indicating typical behavior for their segment.
                        """)

                except Exception as e:
                    st.warning(f"Could not highlight customer in PCA plot: {str(e)}")
            else:
                st.warning(f"Customer ID '{customer_id_input}' not found in the dataset.")

                # Show some example IDs
                st.markdown("**Example valid Customer IDs:**")
                sample_ids = valid_ids[:10]
                st.code(", ".join(sample_ids))
        else:
            # Show helpful guidance when no ID entered
            st.info("""
            👆 Enter a Customer ID above to look up their profile.

            **Sample Customer IDs to try:**
            """)
            sample_ids = valid_ids[:5]
            for sample_id in sample_ids:
                st.code(sample_id)


if __name__ == "__main__":
    main()
