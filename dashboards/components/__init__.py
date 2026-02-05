"""
Dashboard Components

Reusable UI components for Streamlit dashboards.
"""

from .data_loader import (
    load_transactions,
    load_rfm,
    load_customer_features,
    load_clusters,
    load_merged_data,
    get_date_range,
    check_data_files,
    DATA_DIR,
)

from .filters import (
    render_date_filter,
    render_cluster_filter,
    render_k_selector,
    render_top_n_filter,
    apply_date_filter,
    apply_cluster_filter,
)

from .kpi_cards import (
    render_kpi_row,
    calculate_business_kpis,
    format_currency,
    format_number,
)

from .tables import (
    render_top_customers_table,
    render_cluster_summary_table,
    render_customer_lookup_result,
    render_feature_comparison_table,
)

__all__ = [
    # Data loader
    "load_transactions",
    "load_rfm",
    "load_customer_features",
    "load_clusters",
    "load_merged_data",
    "get_date_range",
    "check_data_files",
    "DATA_DIR",
    # Filters
    "render_date_filter",
    "render_cluster_filter",
    "render_k_selector",
    "render_top_n_filter",
    "apply_date_filter",
    "apply_cluster_filter",
    # KPI cards
    "render_kpi_row",
    "calculate_business_kpis",
    "format_currency",
    "format_number",
    # Tables
    "render_top_customers_table",
    "render_cluster_summary_table",
    "render_customer_lookup_result",
    "render_feature_comparison_table",
]
