"""
Visualization functions for Streamlit dashboards.

This package contains Plotly chart functions organized by dashboard:
- business_charts: Charts for Business Analytics Dashboard
- ds_charts: Charts for Data Science Dashboard
"""

from .business_charts import (
    create_daily_revenue_chart,
    create_monthly_revenue_chart,
    create_revenue_by_cluster_chart,
    create_purchase_heatmap,
    create_top_products_chart,
    create_customer_distribution_chart,
    create_cluster_distribution_chart,
    create_rfm_distributions,
)

from .ds_charts import (
    create_feature_boxplots,
    create_feature_histograms,
    create_pca_variance_chart,
    create_pca_2d_scatter,
    create_pca_3d_scatter,
    create_elbow_chart,
    create_silhouette_chart,
    create_cluster_radar_chart,
    create_individual_radar_charts,
)

__all__ = [
    # Business charts
    "create_daily_revenue_chart",
    "create_monthly_revenue_chart",
    "create_revenue_by_cluster_chart",
    "create_purchase_heatmap",
    "create_top_products_chart",
    "create_customer_distribution_chart",
    "create_cluster_distribution_chart",
    "create_rfm_distributions",
    # DS charts
    "create_feature_boxplots",
    "create_feature_histograms",
    "create_pca_variance_chart",
    "create_pca_2d_scatter",
    "create_pca_3d_scatter",
    "create_elbow_chart",
    "create_silhouette_chart",
    "create_cluster_radar_chart",
    "create_individual_radar_charts",
]
