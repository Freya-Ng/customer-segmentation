"""
KPI Cards

KPI metric display components.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Any


def format_currency(value: float) -> str:
    """
    Format number as GBP currency.

    Args:
        value: Numeric value

    Returns:
        Formatted string (e.g., "£1,234,567.89")
    """
    return f"£{value:,.2f}"


def format_number(value: float, decimals: int = 0) -> str:
    """
    Format number with thousands separator.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "1,234,567")
    """
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def calculate_business_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate core business KPIs from transaction data.

    Args:
        df: Transaction DataFrame with CustomerID, InvoiceNo, TotalPrice, Quantity

    Returns:
        Dict with KPI values
    """
    total_revenue = df["TotalPrice"].sum()
    total_customers = df["CustomerID"].nunique()
    total_orders = df["InvoiceNo"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    total_items = df["Quantity"].sum()
    avg_items_per_order = total_items / total_orders if total_orders > 0 else 0

    return {
        "total_revenue": total_revenue,
        "total_customers": total_customers,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "total_items": total_items,
        "avg_items_per_order": avg_items_per_order,
    }


def render_kpi_row(metrics: List[Dict[str, Any]]) -> None:
    """
    Render a row of KPI cards using st.columns and st.metric.

    Args:
        metrics: List of metric dicts with keys:
            - label: Display label
            - value: Numeric value
            - format: "currency", "number", or "decimal"
            - delta: Optional delta value
            - delta_color: "normal", "inverse", or "off"
    """
    cols = st.columns(len(metrics))

    for col, metric in zip(cols, metrics):
        label = metric.get("label", "Metric")
        value = metric.get("value", 0)
        fmt = metric.get("format", "number")
        delta = metric.get("delta")
        delta_color = metric.get("delta_color", "normal")

        # Format the value
        if fmt == "currency":
            display_value = format_currency(value)
        elif fmt == "decimal":
            display_value = format_number(value, decimals=2)
        else:
            display_value = format_number(value)

        # Format delta if present
        display_delta = None
        if delta is not None:
            if fmt == "currency":
                display_delta = format_currency(delta)
            else:
                display_delta = f"{delta:+.1%}" if abs(delta) < 100 else format_number(delta)

        with col:
            st.metric(
                label=label,
                value=display_value,
                delta=display_delta,
                delta_color=delta_color
            )
