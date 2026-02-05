"""
Tables

Data table components for Streamlit dashboards.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional


# Feature names for display
FEATURE_NAMES_EN = {
    "Sum_Quantity": "Total Purchase Quantity",
    "Mean_UnitPrice": "Average Price",
    "Mean_TotalPrice": "Avg Transaction Value",
    "Sum_TotalPrice": "Total Spending",
    "Count_Invoice": "Number of Purchases",
    "Count_Stock": "Number of Different Products",
    "Mean_InvoiceCountPerStock": "Purchase Frequency/Product",
    "Mean_StockCountPerInvoice": "Products/Transaction",
    "Mean_UnitPriceMeanPerInvoice": "Avg Price/Transaction",
    "Mean_QuantitySumPerInvoice": "Quantity/Transaction",
    "Mean_TotalPriceMeanPerInvoice": "Avg Value/Transaction",
    "Mean_TotalPriceSumPerInvoice": "Total Value/Transaction",
    "Mean_UnitPriceMeanPerStock": "Avg Price/Product",
    "Mean_QuantitySumPerStock": "Avg Quantity/Product",
    "Mean_TotalPriceMeanPerStock": "Avg Value/Product",
    "Mean_TotalPriceSumPerStock": "Total Value/Product",
}


def render_top_customers_table(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Display top customers by revenue.

    Args:
        df: Transaction DataFrame with CustomerID, TotalPrice, InvoiceNo
        top_n: Number of customers to show
    """
    # Aggregate by customer
    customer_stats = df.groupby("CustomerID").agg({
        "TotalPrice": "sum",
        "InvoiceNo": "nunique",
        "Quantity": "sum"
    }).reset_index()

    customer_stats.columns = ["Customer ID", "Total Revenue", "Orders", "Items"]
    customer_stats["Avg Order Value"] = customer_stats["Total Revenue"] / customer_stats["Orders"]

    # Sort and get top N
    customer_stats = customer_stats.nlargest(top_n, "Total Revenue").reset_index(drop=True)
    customer_stats.index = customer_stats.index + 1  # 1-based ranking
    customer_stats.index.name = "Rank"

    # Display with formatting
    st.dataframe(
        customer_stats,
        column_config={
            "Customer ID": st.column_config.TextColumn("Customer ID"),
            "Total Revenue": st.column_config.NumberColumn(
                "Total Revenue",
                format="£%.2f"
            ),
            "Orders": st.column_config.NumberColumn("Orders", format="%d"),
            "Items": st.column_config.NumberColumn("Items", format="%d"),
            "Avg Order Value": st.column_config.NumberColumn(
                "Avg Order Value",
                format="£%.2f"
            ),
        },
        use_container_width=True
    )


def render_cluster_summary_table(
    cluster_means: pd.DataFrame,
    cluster_names: Optional[Dict[int, str]] = None
) -> None:
    """
    Display cluster feature comparison table.

    Args:
        cluster_means: DataFrame with clusters as rows, features as columns
        cluster_names: Optional dict mapping cluster index to name
    """
    # Make a copy and rename index
    display_df = cluster_means.copy()

    if cluster_names:
        display_df.index = [cluster_names.get(i, f"Cluster {i}") for i in display_df.index]
    else:
        display_df.index = [f"Cluster {i}" for i in display_df.index]

    # Rename columns to readable names
    display_df.columns = [FEATURE_NAMES_EN.get(col, col) for col in display_df.columns]

    # Transpose for better readability (features as rows)
    display_df = display_df.T

    st.dataframe(
        display_df.style.format("{:.2f}").background_gradient(axis=1, cmap="RdYlGn"),
        use_container_width=True
    )


def render_customer_lookup_result(
    customer_id: str,
    customer_data: Dict[str, Any]
) -> None:
    """
    Display single customer profile card.

    Args:
        customer_id: Customer identifier
        customer_data: Dict with customer attributes
    """
    st.subheader(f"Customer Profile: {customer_id}")

    # Display cluster info prominently
    if "Cluster" in customer_data:
        cluster = customer_data["Cluster"]
        st.info(f"**Cluster Assignment:** Cluster {cluster}")

    # Create columns for key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        if "Sum_TotalPrice" in customer_data:
            st.metric("Total Spending", f"£{customer_data['Sum_TotalPrice']:,.2f}")
        elif "Monetary" in customer_data:
            st.metric("Total Spending", f"£{customer_data['Monetary']:,.2f}")

    with col2:
        if "Count_Invoice" in customer_data:
            st.metric("Purchases", f"{customer_data['Count_Invoice']:.0f}")
        elif "Frequency" in customer_data:
            st.metric("Purchases", f"{customer_data['Frequency']:.0f}")

    with col3:
        if "Count_Stock" in customer_data:
            st.metric("Products", f"{customer_data['Count_Stock']:.0f}")
        elif "Recency" in customer_data:
            st.metric("Recency (Days)", f"{customer_data['Recency']:.0f}")

    # Show all features in expandable section
    with st.expander("View All Features"):
        features_df = pd.DataFrame([customer_data])
        features_df = features_df.T.reset_index()
        features_df.columns = ["Feature", "Value"]
        features_df["Feature"] = features_df["Feature"].map(
            lambda x: FEATURE_NAMES_EN.get(x, x)
        )
        st.dataframe(features_df, use_container_width=True, hide_index=True)


def render_feature_comparison_table(
    features_df: pd.DataFrame,
    cluster_col: str = "Cluster"
) -> None:
    """
    Display feature means by cluster with conditional formatting.

    Args:
        features_df: DataFrame with features and cluster column
        cluster_col: Name of cluster column
    """
    # Calculate means by cluster
    feature_cols = [col for col in features_df.columns
                    if col not in [cluster_col, "CustomerID"]]

    cluster_means = features_df.groupby(cluster_col)[feature_cols].mean()

    # Rename columns
    cluster_means.columns = [FEATURE_NAMES_EN.get(col, col) for col in cluster_means.columns]

    # Rename index
    cluster_means.index = [f"Cluster {i}" for i in cluster_means.index]

    # Display with gradient coloring
    st.dataframe(
        cluster_means.style.format("{:.2f}").background_gradient(axis=0, cmap="RdYlGn"),
        use_container_width=True
    )
