"""
Data Science Charts

Plotly chart functions for the Data Science Dashboard.
All functions accept DataFrames/arrays as input and return go.Figure objects.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# English feature names mapping
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

# Features for radar charts
RADAR_FEATURES = [
    "Sum_Quantity",
    "Sum_TotalPrice",
    "Mean_UnitPrice",
    "Count_Invoice",
    "Count_Stock",
    "Mean_TotalPriceSumPerInvoice",
]

RADAR_LABELS = [
    "Purchase Volume",
    "Total Spending",
    "Preferred Price",
    "Purchase Frequency",
    "Product Diversity",
    "Value/Transaction",
]

# Color palette for clusters
CLUSTER_COLORS = {
    0: "#E74C3C",  # Red
    1: "#3498DB",  # Blue
    2: "#2ECC71",  # Green
    3: "#F39C12",  # Orange
    4: "#9B59B6",  # Purple
}


def create_feature_boxplots(features_df: pd.DataFrame, transformed: bool = False) -> go.Figure:
    """
    Create a 4x4 grid of boxplots for all 16 features.

    Args:
        features_df: DataFrame with 16 feature columns
        transformed: Whether showing transformed (Box-Cox) features

    Returns:
        Plotly Figure object
    """
    # Get feature columns (exclude CustomerID if present)
    feature_cols = [col for col in features_df.columns if col != "CustomerID" and col in FEATURE_NAMES_EN]

    if len(feature_cols) == 0:
        feature_cols = [col for col in features_df.columns if col != "CustomerID"]

    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[FEATURE_NAMES_EN.get(col, col)[:20] for col in feature_cols],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    for i, col in enumerate(feature_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        fig.add_trace(
            go.Box(
                y=features_df[col],
                name=col,
                marker_color="#3498DB",
                showlegend=False,
                hovertemplate=f"{FEATURE_NAMES_EN.get(col, col)}<br>Value: %{{y:.2f}}<extra></extra>"
            ),
            row=row, col=col_idx
        )

    title = "Feature Distributions (Box-Cox Transformed)" if transformed else "Feature Distributions (Original)"
    fig.update_layout(
        title=title,
        showlegend=False,
        template="plotly_white",
        height=200 * n_rows
    )

    return fig


def create_feature_histograms(features_df: pd.DataFrame, transformed: bool = False) -> go.Figure:
    """
    Create a 4x4 grid of histograms for all 16 features.

    Args:
        features_df: DataFrame with 16 feature columns
        transformed: Whether showing transformed (Box-Cox) features

    Returns:
        Plotly Figure object
    """
    # Get feature columns (exclude CustomerID if present)
    feature_cols = [col for col in features_df.columns if col != "CustomerID" and col in FEATURE_NAMES_EN]

    if len(feature_cols) == 0:
        feature_cols = [col for col in features_df.columns if col != "CustomerID"]

    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[FEATURE_NAMES_EN.get(col, col)[:20] for col in feature_cols],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    for i, col in enumerate(feature_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        fig.add_trace(
            go.Histogram(
                x=features_df[col],
                nbinsx=30,
                marker_color="#3498DB",
                opacity=0.7,
                showlegend=False,
                hovertemplate=f"{FEATURE_NAMES_EN.get(col, col)}<br>Value: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"
            ),
            row=row, col=col_idx
        )

    title = "Feature Histograms (Box-Cox Transformed)" if transformed else "Feature Histograms (Original)"
    fig.update_layout(
        title=title,
        showlegend=False,
        template="plotly_white",
        height=200 * n_rows
    )

    return fig


def create_pca_variance_chart(explained_variance_ratio: np.ndarray) -> go.Figure:
    """
    Create a bar + line chart for PCA explained variance.

    Args:
        explained_variance_ratio: Array of explained variance ratios from PCA

    Returns:
        Plotly Figure object
    """
    n_components = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    fig = go.Figure()

    # Individual variance bars
    fig.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(n_components)],
        y=explained_variance_ratio,
        name="Individual",
        marker_color="#3498DB",
        hovertemplate="Component: %{x}<br>Variance: %{y:.2%}<extra></extra>"
    ))

    # Cumulative line
    fig.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(n_components)],
        y=cumulative,
        mode="lines+markers",
        name="Cumulative",
        line=dict(color="#E74C3C", width=2),
        marker=dict(size=8),
        hovertemplate="Component: %{x}<br>Cumulative: %{y:.2%}<extra></extra>"
    ))

    # Add threshold lines
    fig.add_hline(y=0.8, line_dash="dash", line_color="gray",
                  annotation_text="80%", annotation_position="right")
    fig.add_hline(y=0.9, line_dash="dash", line_color="gray",
                  annotation_text="90%", annotation_position="right")

    fig.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
        yaxis_tickformat=".0%",
        template="plotly_white",
        legend=dict(yanchor="middle", y=0.5, xanchor="right", x=0.99)
    )

    return fig


def create_pca_2d_scatter(
    pca_df: pd.DataFrame,
    cluster_col: str = "Cluster"
) -> go.Figure:
    """
    Create a 2D scatter plot on PC1/PC2 colored by cluster.

    Args:
        pca_df: DataFrame with PC1, PC2, and Cluster columns
        cluster_col: Name of cluster column

    Returns:
        Plotly Figure object
    """
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=cluster_col,
        color_discrete_map={i: CLUSTER_COLORS.get(i, "#95A5A6") for i in pca_df[cluster_col].unique()},
        title="PCA 2D Visualization",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        template="plotly_white",
        opacity=0.6
    )

    fig.update_traces(marker=dict(size=6))
    fig.update_layout(legend_title_text="Cluster")

    return fig


def create_pca_3d_scatter(
    pca_df: pd.DataFrame,
    cluster_col: str = "Cluster"
) -> go.Figure:
    """
    Create a 3D scatter plot on PC1/PC2/PC3 colored by cluster.

    Args:
        pca_df: DataFrame with PC1, PC2, PC3, and Cluster columns
        cluster_col: Name of cluster column

    Returns:
        Plotly Figure object
    """
    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color=cluster_col,
        color_discrete_map={i: CLUSTER_COLORS.get(i, "#95A5A6") for i in pca_df[cluster_col].unique()},
        title="PCA 3D Visualization",
        labels={
            "PC1": "PC1",
            "PC2": "PC2",
            "PC3": "PC3"
        },
        template="plotly_white",
        opacity=0.6
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        legend_title_text="Cluster",
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3"
        )
    )

    return fig


def create_elbow_chart(k_range: list, inertias: list) -> go.Figure:
    """
    Create an elbow method line chart.

    Args:
        k_range: List of k values tested
        inertias: List of inertia values for each k

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_range,
        y=inertias,
        mode="lines+markers",
        name="Inertia",
        line=dict(color="#3498DB", width=2),
        marker=dict(size=10),
        hovertemplate="k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title="Elbow Method",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia (Within-cluster Sum of Squares)",
        template="plotly_white",
        xaxis=dict(dtick=1)
    )

    return fig


def create_silhouette_chart(
    k_range: list,
    silhouette_scores: list,
    best_k: int = None
) -> go.Figure:
    """
    Create a silhouette score line chart with best k highlighted.

    Args:
        k_range: List of k values tested
        silhouette_scores: List of silhouette scores for each k
        best_k: Optimal k value to highlight

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_range,
        y=silhouette_scores,
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#2ECC71", width=2),
        marker=dict(size=10),
        hovertemplate="k=%{x}<br>Score: %{y:.4f}<extra></extra>"
    ))

    # Highlight best k
    if best_k is not None and best_k in k_range:
        idx = k_range.index(best_k) if isinstance(k_range, list) else list(k_range).index(best_k)
        fig.add_trace(go.Scatter(
            x=[best_k],
            y=[silhouette_scores[idx]],
            mode="markers",
            name=f"Best k={best_k}",
            marker=dict(size=15, color="#E74C3C", symbol="star"),
            hovertemplate=f"Best k={best_k}<br>Score: {silhouette_scores[idx]:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title="Silhouette Score",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        template="plotly_white",
        xaxis=dict(dtick=1)
    )

    return fig


def create_cluster_radar_chart(
    cluster_means: pd.DataFrame,
    cluster_names: dict = None,
    features: list = None
) -> go.Figure:
    """
    Create an overlaid radar chart for all clusters.

    Args:
        cluster_means: DataFrame with clusters as rows, features as columns
        cluster_names: Dict mapping cluster index to name
        features: List of features to include (uses RADAR_FEATURES by default)

    Returns:
        Plotly Figure object
    """
    if features is None:
        features = [f for f in RADAR_FEATURES if f in cluster_means.columns]
        labels = [RADAR_LABELS[RADAR_FEATURES.index(f)] for f in features]
    else:
        labels = [FEATURE_NAMES_EN.get(f, f) for f in features]

    if len(features) == 0:
        # Fallback: use first 6 columns
        features = list(cluster_means.columns)[:6]
        labels = features

    # Normalize data for radar (0-1 scale)
    data_for_radar = cluster_means[features].copy()
    for col in data_for_radar.columns:
        min_val = data_for_radar[col].min()
        max_val = data_for_radar[col].max()
        if max_val > min_val:
            data_for_radar[col] = (data_for_radar[col] - min_val) / (max_val - min_val)
        else:
            data_for_radar[col] = 0.5

    fig = go.Figure()

    for cluster_idx in data_for_radar.index:
        values = data_for_radar.loc[cluster_idx].values.tolist()
        values.append(values[0])  # Close the polygon

        name = cluster_names.get(cluster_idx, f"Cluster {cluster_idx}") if cluster_names else f"Cluster {cluster_idx}"
        color = CLUSTER_COLORS.get(cluster_idx, "#95A5A6")

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=name,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.3,
            hovertemplate=f"{name}<br>%{{theta}}: %{{r:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0%", "25%", "50%", "75%", "100%"]
            )
        ),
        title="Cluster Profiles (Radar Chart)",
        template="plotly_white",
        legend=dict(yanchor="top", y=1.1, xanchor="center", x=0.5, orientation="h")
    )

    return fig


def create_individual_radar_charts(
    cluster_means: pd.DataFrame,
    cluster_names: dict = None,
    features: list = None
) -> go.Figure:
    """
    Create individual radar subplots per cluster.

    Args:
        cluster_means: DataFrame with clusters as rows, features as columns
        cluster_names: Dict mapping cluster index to name
        features: List of features to include

    Returns:
        Plotly Figure object
    """
    if features is None:
        features = [f for f in RADAR_FEATURES if f in cluster_means.columns]
        labels = [RADAR_LABELS[RADAR_FEATURES.index(f)] for f in features]
    else:
        labels = [FEATURE_NAMES_EN.get(f, f) for f in features]

    if len(features) == 0:
        features = list(cluster_means.columns)[:6]
        labels = features

    n_clusters = len(cluster_means)
    n_cols = min(2, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    # Normalize data
    data_for_radar = cluster_means[features].copy()
    for col in data_for_radar.columns:
        min_val = data_for_radar[col].min()
        max_val = data_for_radar[col].max()
        if max_val > min_val:
            data_for_radar[col] = (data_for_radar[col] - min_val) / (max_val - min_val)
        else:
            data_for_radar[col] = 0.5

    # Create subplot specs for polar charts
    specs = [[{"type": "polar"} for _ in range(n_cols)] for _ in range(n_rows)]
    subplot_titles = [
        cluster_names.get(idx, f"Cluster {idx}") if cluster_names else f"Cluster {idx}"
        for idx in cluster_means.index
    ]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles
    )

    for i, cluster_idx in enumerate(data_for_radar.index):
        row = i // n_cols + 1
        col = i % n_cols + 1

        values = data_for_radar.loc[cluster_idx].values.tolist()
        values.append(values[0])

        color = CLUSTER_COLORS.get(cluster_idx, "#95A5A6")

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels + [labels[0]],
                fill="toself",
                name=f"Cluster {cluster_idx}",
                line=dict(color=color, width=2),
                fillcolor=color,
                opacity=0.3,
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        title="Individual Cluster Profiles",
        template="plotly_white",
        height=350 * n_rows
    )

    # Update polar axis for all subplots
    for i in range(n_clusters):
        polar_key = f"polar{i+1}" if i > 0 else "polar"
        fig.update_layout(**{
            polar_key: dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.5, 1.0],
                    ticktext=["0%", "50%", "100%"]
                )
            )
        })

    return fig
