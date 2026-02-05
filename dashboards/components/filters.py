"""
Filters

Sidebar filter widgets for Streamlit dashboards.
"""

import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Tuple


def render_date_filter(
    min_date: datetime,
    max_date: datetime,
    key: str = "date_filter"
) -> Tuple[datetime, datetime]:
    """
    Render date range picker in sidebar.

    Args:
        min_date: Minimum date in data
        max_date: Maximum date in data
        key: Unique key for the widget

    Returns:
        Tuple of (start_date, end_date)
    """
    st.sidebar.subheader("Date Range")

    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key=f"{key}_start"
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key=f"{key}_end"
    )

    return (
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.max.time())
    )


def render_cluster_filter(
    available_clusters: List[int],
    key: str = "cluster_filter"
) -> List[int]:
    """
    Render cluster multi-select in sidebar.

    Args:
        available_clusters: List of available cluster IDs
        key: Unique key for the widget

    Returns:
        List of selected cluster IDs
    """
    st.sidebar.subheader("Clusters")

    selected = st.sidebar.multiselect(
        "Select Clusters",
        options=available_clusters,
        default=available_clusters,
        format_func=lambda x: f"Cluster {x}",
        key=key
    )

    return selected if selected else available_clusters


def render_k_selector(
    available_k: List[int] = None,
    key: str = "k_selector"
) -> int:
    """
    Render cluster count selector.

    Args:
        available_k: List of available k values
        key: Unique key for the widget

    Returns:
        Selected k value
    """
    if available_k is None:
        available_k = [3, 4]

    st.sidebar.subheader("Cluster Settings")

    k = st.sidebar.radio(
        "Number of Clusters (k)",
        options=available_k,
        index=0,
        key=key
    )

    return k


def render_top_n_filter(
    default: int = 10,
    max_val: int = 50,
    key: str = "top_n"
) -> int:
    """
    Render top N slider for rankings.

    Args:
        default: Default value
        max_val: Maximum value
        key: Unique key for the widget

    Returns:
        Selected N value
    """
    n = st.sidebar.slider(
        "Top N Items",
        min_value=5,
        max_value=max_val,
        value=default,
        step=5,
        key=key
    )

    return n


def apply_date_filter(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    date_column: str = "InvoiceDate"
) -> pd.DataFrame:
    """
    Filter dataframe by date range.

    Args:
        df: Input DataFrame
        start_date: Start date
        end_date: End date
        date_column: Name of date column

    Returns:
        Filtered DataFrame
    """
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df[mask]


def apply_cluster_filter(
    df: pd.DataFrame,
    selected_clusters: List[int],
    cluster_column: str = "Cluster"
) -> pd.DataFrame:
    """
    Filter dataframe by selected clusters.

    Args:
        df: Input DataFrame
        selected_clusters: List of cluster IDs to include
        cluster_column: Name of cluster column

    Returns:
        Filtered DataFrame
    """
    if cluster_column not in df.columns:
        return df

    return df[df[cluster_column].isin(selected_clusters)]
