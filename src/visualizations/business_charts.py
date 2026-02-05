"""
Business Analytics Charts

Plotly chart functions for the Business Analytics Dashboard.
All functions accept DataFrames as input and return go.Figure objects.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color palette for clusters
CLUSTER_COLORS = {
    0: "#E74C3C",  # Red
    1: "#3498DB",  # Blue
    2: "#2ECC71",  # Green
    3: "#F39C12",  # Orange
    4: "#9B59B6",  # Purple
}

# Day names for heatmap
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def create_daily_revenue_chart(df: pd.DataFrame, show_trend: bool = True) -> go.Figure:
    """
    Create a line chart showing daily revenue with optional moving average.

    Args:
        df: DataFrame with InvoiceDate and TotalPrice columns
        show_trend: Whether to show 7-day moving average

    Returns:
        Plotly Figure object
    """
    # Group by date
    daily = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
    daily.columns = ["Date", "Revenue"]
    daily["Date"] = pd.to_datetime(daily["Date"])

    fig = go.Figure()

    # Daily revenue line
    fig.add_trace(go.Scatter(
        x=daily["Date"],
        y=daily["Revenue"],
        mode="lines",
        name="Daily Revenue",
        line=dict(color="#3498DB", width=1),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Revenue: £%{y:,.2f}<extra></extra>"
    ))

    # Add moving average
    if show_trend and len(daily) >= 7:
        daily["MA7"] = daily["Revenue"].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=daily["Date"],
            y=daily["MA7"],
            mode="lines",
            name="7-Day Moving Avg",
            line=dict(color="#E74C3C", width=2, dash="dash"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>7-Day Avg: £%{y:,.2f}<extra></extra>"
        ))

    fig.update_layout(
        title="Daily Revenue Trend",
        xaxis_title="Date",
        yaxis_title="Revenue (£)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white"
    )

    return fig


def create_monthly_revenue_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing monthly revenue.

    Args:
        df: DataFrame with InvoiceDate and TotalPrice columns

    Returns:
        Plotly Figure object
    """
    # Group by month
    df = df.copy()
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = df.groupby("Month")["TotalPrice"].sum().reset_index()
    monthly.columns = ["Month", "Revenue"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthly["Month"],
        y=monthly["Revenue"],
        marker_color="#3498DB",
        hovertemplate="Month: %{x}<br>Revenue: £%{y:,.2f}<extra></extra>"
    ))

    # Add average line
    avg_revenue = monthly["Revenue"].mean()
    fig.add_hline(
        y=avg_revenue,
        line_dash="dash",
        line_color="#E74C3C",
        annotation_text=f"Avg: £{avg_revenue:,.0f}",
        annotation_position="right"
    )

    fig.update_layout(
        title="Monthly Revenue",
        xaxis_title="Month",
        yaxis_title="Revenue (£)",
        template="plotly_white"
    )

    return fig


def create_revenue_by_cluster_chart(
    df: pd.DataFrame,
    chart_type: str = "pie",
    cluster_col: str = "Cluster"
) -> go.Figure:
    """
    Create a pie or bar chart showing revenue distribution by cluster.

    Args:
        df: DataFrame with Cluster and TotalPrice columns
        chart_type: "pie" or "bar"
        cluster_col: Name of cluster column

    Returns:
        Plotly Figure object
    """
    # Calculate revenue by cluster
    cluster_revenue = df.groupby(cluster_col)["TotalPrice"].sum().reset_index()
    cluster_revenue.columns = ["Cluster", "Revenue"]
    cluster_revenue["Cluster"] = cluster_revenue["Cluster"].astype(str)

    colors = [CLUSTER_COLORS.get(i, "#95A5A6") for i in range(len(cluster_revenue))]

    if chart_type == "pie":
        fig = go.Figure(data=[go.Pie(
            labels=cluster_revenue["Cluster"],
            values=cluster_revenue["Revenue"],
            marker_colors=colors,
            textinfo="percent+label",
            hovertemplate="Cluster %{label}<br>Revenue: £%{value:,.2f}<br>Share: %{percent}<extra></extra>"
        )])
        fig.update_layout(title="Revenue by Cluster")
    else:
        fig = go.Figure(data=[go.Bar(
            x=cluster_revenue["Cluster"],
            y=cluster_revenue["Revenue"],
            marker_color=colors,
            hovertemplate="Cluster %{x}<br>Revenue: £%{y:,.2f}<extra></extra>"
        )])
        fig.update_layout(
            title="Revenue by Cluster",
            xaxis_title="Cluster",
            yaxis_title="Revenue (£)"
        )

    fig.update_layout(template="plotly_white")
    return fig


def create_purchase_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing purchase activity by day of week and hour.

    Args:
        df: DataFrame with DayOfWeek (0-6) and HourOfDay (0-23) columns

    Returns:
        Plotly Figure object
    """
    # Create pivot table
    heatmap_data = df.groupby(["DayOfWeek", "HourOfDay"]).size().reset_index(name="Count")
    pivot = heatmap_data.pivot(index="DayOfWeek", columns="HourOfDay", values="Count").fillna(0)

    # Ensure all hours are present
    for hour in range(24):
        if hour not in pivot.columns:
            pivot[hour] = 0
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Ensure all days are present
    for day in range(7):
        if day not in pivot.index:
            pivot.loc[day] = 0
    pivot = pivot.sort_index()

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[DAY_NAMES[d] for d in pivot.index],
        colorscale="Viridis",
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Transactions: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title="Purchase Activity by Day and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template="plotly_white"
    )

    return fig


def create_top_products_chart(
    df: pd.DataFrame,
    top_n: int = 10,
    metric: str = "quantity"
) -> go.Figure:
    """
    Create a horizontal bar chart of top products.

    Args:
        df: DataFrame with Description, Quantity, and TotalPrice columns
        top_n: Number of products to show
        metric: "quantity" or "revenue"

    Returns:
        Plotly Figure object
    """
    if metric == "quantity":
        product_data = df.groupby("Description")["Quantity"].sum().nlargest(top_n).reset_index()
        product_data.columns = ["Product", "Value"]
        title = f"Top {top_n} Products by Quantity"
        value_label = "Quantity"
        hover_template = "Product: %{y}<br>Quantity: %{x:,.0f}<extra></extra>"
    else:
        product_data = df.groupby("Description")["TotalPrice"].sum().nlargest(top_n).reset_index()
        product_data.columns = ["Product", "Value"]
        title = f"Top {top_n} Products by Revenue"
        value_label = "Revenue (£)"
        hover_template = "Product: %{y}<br>Revenue: £%{x:,.2f}<extra></extra>"

    # Sort for proper display (ascending so largest is at top)
    product_data = product_data.sort_values("Value", ascending=True)

    fig = go.Figure(data=[go.Bar(
        x=product_data["Value"],
        y=product_data["Product"],
        orientation="h",
        marker_color="#3498DB",
        hovertemplate=hover_template
    )])

    fig.update_layout(
        title=title,
        xaxis_title=value_label,
        yaxis_title="",
        template="plotly_white",
        height=max(400, top_n * 35)
    )

    return fig


def create_customer_distribution_chart(
    df: pd.DataFrame,
    metric: str = "transactions"
) -> go.Figure:
    """
    Create a histogram showing customer behavior distribution.

    Args:
        df: DataFrame with CustomerID, InvoiceNo, and TotalPrice columns
        metric: "transactions" or "spending"

    Returns:
        Plotly Figure object
    """
    if metric == "transactions":
        # Transactions per customer
        customer_data = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
        customer_data.columns = ["CustomerID", "Value"]
        title = "Distribution of Transactions per Customer"
        xaxis_title = "Number of Transactions"
    else:
        # Spending per customer
        customer_data = df.groupby("CustomerID")["TotalPrice"].sum().reset_index()
        customer_data.columns = ["CustomerID", "Value"]
        # Filter outliers (99th percentile)
        threshold = customer_data["Value"].quantile(0.99)
        customer_data = customer_data[customer_data["Value"] <= threshold]
        title = "Distribution of Customer Spending"
        xaxis_title = "Total Spending (£)"

    fig = go.Figure(data=[go.Histogram(
        x=customer_data["Value"],
        nbinsx=50,
        marker_color="#3498DB",
        opacity=0.7,
        hovertemplate="Range: %{x}<br>Customers: %{y}<extra></extra>"
    )])

    # Add mean line
    mean_val = customer_data["Value"].mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#E74C3C",
        annotation_text=f"Mean: {mean_val:,.1f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Number of Customers",
        template="plotly_white"
    )

    return fig


def create_cluster_distribution_chart(
    df: pd.DataFrame,
    cluster_col: str = "Cluster"
) -> go.Figure:
    """
    Create a pie chart showing customer distribution by cluster.

    Args:
        df: DataFrame with Cluster column
        cluster_col: Name of cluster column

    Returns:
        Plotly Figure object
    """
    # Calculate customer count by cluster
    cluster_counts = df.groupby(cluster_col)["CustomerID"].nunique().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    cluster_counts["Cluster"] = cluster_counts["Cluster"].astype(str)

    colors = [CLUSTER_COLORS.get(i, "#95A5A6") for i in range(len(cluster_counts))]

    fig = go.Figure(data=[go.Pie(
        labels=cluster_counts["Cluster"],
        values=cluster_counts["Count"],
        marker_colors=colors,
        textinfo="percent+label",
        hovertemplate="Cluster %{label}<br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title="Customer Distribution by Cluster",
        template="plotly_white"
    )

    return fig


def create_rfm_distributions(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Create subplots showing R, F, M distributions.

    Args:
        rfm_df: DataFrame with Recency, Frequency, Monetary columns

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency (Days)", "Frequency (Transactions)", "Monetary (£)")
    )

    # Recency
    fig.add_trace(
        go.Histogram(
            x=rfm_df["Recency"],
            nbinsx=30,
            marker_color="#E74C3C",
            opacity=0.7,
            name="Recency",
            hovertemplate="Days: %{x}<br>Customers: %{y}<extra></extra>"
        ),
        row=1, col=1
    )

    # Frequency
    fig.add_trace(
        go.Histogram(
            x=rfm_df["Frequency"],
            nbinsx=30,
            marker_color="#3498DB",
            opacity=0.7,
            name="Frequency",
            hovertemplate="Transactions: %{x}<br>Customers: %{y}<extra></extra>"
        ),
        row=1, col=2
    )

    # Monetary (filter outliers)
    monetary_filtered = rfm_df["Monetary"][rfm_df["Monetary"] <= rfm_df["Monetary"].quantile(0.99)]
    fig.add_trace(
        go.Histogram(
            x=monetary_filtered,
            nbinsx=30,
            marker_color="#2ECC71",
            opacity=0.7,
            name="Monetary",
            hovertemplate="Spending: £%{x:,.0f}<br>Customers: %{y}<extra></extra>"
        ),
        row=1, col=3
    )

    fig.update_layout(
        title="RFM Distributions",
        showlegend=False,
        template="plotly_white",
        height=400
    )

    # Update axis labels
    fig.update_xaxes(title_text="Days Since Last Purchase", row=1, col=1)
    fig.update_xaxes(title_text="Number of Transactions", row=1, col=2)
    fig.update_xaxes(title_text="Total Spending (£)", row=1, col=3)
    fig.update_yaxes(title_text="Customers", row=1, col=1)

    return fig
