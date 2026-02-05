"""
Business Analytics Dashboard

Dashboard for business stakeholders with revenue, customer, and product insights.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboards.components.data_loader import (
    load_transactions, load_rfm, load_clusters, load_merged_data, get_date_range
)
from dashboards.components.filters import (
    render_date_filter, render_cluster_filter, render_top_n_filter,
    apply_date_filter, apply_cluster_filter
)
from dashboards.components.kpi_cards import render_kpi_row, calculate_business_kpis
from dashboards.components.tables import render_top_customers_table

from src.visualizations.business_charts import (
    create_daily_revenue_chart,
    create_monthly_revenue_chart,
    create_revenue_by_cluster_chart,
    create_purchase_heatmap,
    create_top_products_chart,
    create_customer_distribution_chart,
    create_cluster_distribution_chart,
    create_rfm_distributions
)


def load_css():
    """Load custom CSS styling."""
    css_path = Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_kpi_insights(kpis: dict, filtered_df) -> dict:
    """Generate insights based on KPI values."""
    insights = {}

    # AOV insight
    if kpis["avg_order_value"] > 500:
        insights["aov"] = ("high", "High AOV indicates premium customers. Focus on retention strategies.")
    elif kpis["avg_order_value"] > 200:
        insights["aov"] = ("medium", "Healthy AOV. Consider upselling to increase basket size.")
    else:
        insights["aov"] = ("low", "Low AOV suggests opportunity for bundling or cross-selling.")

    # Customer concentration
    top_10_pct = filtered_df.groupby("CustomerID")["TotalPrice"].sum().nlargest(10).sum() / kpis["total_revenue"] * 100
    if top_10_pct > 50:
        insights["concentration"] = ("warning", f"Top 10 customers = {top_10_pct:.1f}% revenue. High dependency risk!")
    else:
        insights["concentration"] = ("good", f"Top 10 customers = {top_10_pct:.1f}% revenue. Well-diversified.")

    return insights


def main():
    # Page configuration
    st.set_page_config(
        page_title="Business Analytics",
        page_icon="📊",
        layout="wide"
    )

    # Load custom CSS
    load_css()

    # Header
    st.title("📊 Business Analytics Dashboard")
    st.markdown("*Actionable insights for business decision-making*")

    # Load data
    try:
        transactions = load_transactions()
        rfm = load_rfm()
        merged_df = load_merged_data(k=3)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure all data files exist in `data/processed/`")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date filter
    min_date, max_date = get_date_range()
    start_date, end_date = render_date_filter(min_date, max_date)

    # Cluster filter
    available_clusters = sorted(merged_df["Cluster"].dropna().unique().astype(int).tolist())
    selected_clusters = render_cluster_filter(available_clusters)

    # Apply filters
    filtered_df = apply_date_filter(merged_df, start_date, end_date)
    filtered_df = apply_cluster_filter(filtered_df, selected_clusters)

    # Show filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Data:**")
    st.sidebar.write(f"- Transactions: {len(filtered_df):,}")
    st.sidebar.write(f"- Customers: {filtered_df['CustomerID'].nunique():,}")

    # ============ KPI Section ============
    st.header("Key Performance Indicators")

    with st.expander("ℹ️ **Understanding KPIs** - Click to learn what these metrics mean", expanded=False):
        st.markdown("""
        | KPI | What It Measures | Why It Matters | Action If Low | Action If High |
        |-----|------------------|----------------|---------------|----------------|
        | **Total Revenue** | Sum of all sales | Overall business health | Increase marketing, promotions | Maintain momentum, expand |
        | **Total Customers** | Unique buyers | Market reach | Acquisition campaigns | Focus on retention |
        | **Total Orders** | Number of transactions | Purchase frequency | Encourage repeat purchases | Optimize fulfillment |
        | **Avg Order Value (AOV)** | Revenue ÷ Orders | Customer spending behavior | Upsell, cross-sell, bundles | VIP programs, premium products |
        """)

    kpis = calculate_business_kpis(filtered_df)
    insights = get_kpi_insights(kpis, filtered_df)

    render_kpi_row([
        {"label": "Total Revenue", "value": kpis["total_revenue"], "format": "currency"},
        {"label": "Total Customers", "value": kpis["total_customers"], "format": "number"},
        {"label": "Total Orders", "value": kpis["total_orders"], "format": "number"},
        {"label": "Avg Order Value", "value": kpis["avg_order_value"], "format": "currency"},
    ])

    # KPI Insights
    col1, col2 = st.columns(2)
    with col1:
        level, msg = insights["aov"]
        if level == "high":
            st.success(f"💎 **AOV Insight:** {msg}")
        elif level == "medium":
            st.info(f"📈 **AOV Insight:** {msg}")
        else:
            st.warning(f"💡 **AOV Insight:** {msg}")

    with col2:
        level, msg = insights["concentration"]
        if level == "warning":
            st.warning(f"⚠️ **Risk Alert:** {msg}")
        else:
            st.success(f"✅ **Health Check:** {msg}")

    st.markdown("---")

    # ============ Revenue Section ============
    st.header("Revenue Analysis")

    with st.expander("ℹ️ **How to Read Revenue Charts** - Identify trends and seasonality"):
        st.markdown("""
        **Daily Trend Chart:**
        - 📈 **Upward trend**: Business is growing - maintain current strategies
        - 📉 **Downward trend**: Investigate causes - competition, seasonality, or product issues
        - 🔄 **7-Day Moving Average** (red dashed line): Smooths out daily noise to show true trend
        - ⚡ **Spikes**: Likely promotional events or holidays - replicate successful campaigns

        **Monthly Summary Chart:**
        - Compare months to identify **seasonal patterns**
        - **Red line** shows average - months below need attention
        - Use this to **plan inventory and staffing**

        **Business Actions:**
        - If December is highest → Plan holiday promotions early
        - If summer is low → Consider summer sale campaigns
        - Consistent growth → Invest in capacity expansion
        """)

    tab1, tab2 = st.tabs(["📈 Daily Trend", "📊 Monthly Summary"])

    with tab1:
        fig = create_daily_revenue_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

        # Calculate trend insight
        daily_rev = filtered_df.groupby(filtered_df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        if len(daily_rev) > 14:
            first_week = daily_rev.head(7).mean()
            last_week = daily_rev.tail(7).mean()
            change_pct = ((last_week - first_week) / first_week) * 100 if first_week > 0 else 0

            if change_pct > 10:
                st.success(f"📈 **Trend:** Revenue increased by {change_pct:.1f}% (comparing first vs last week)")
            elif change_pct < -10:
                st.error(f"📉 **Trend:** Revenue decreased by {abs(change_pct):.1f}% - Investigate and take action!")
            else:
                st.info(f"➡️ **Trend:** Revenue is stable ({change_pct:+.1f}% change)")

    with tab2:
        fig = create_monthly_revenue_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ============ Cluster Analysis Section ============
    st.header("Customer Segment Analysis")

    with st.expander("ℹ️ **Understanding Customer Segments** - Who are your customers?"):
        st.markdown("""
        Customers are grouped into **clusters** based on their purchasing behavior. Each cluster represents a distinct customer type:

        | Cluster | Typical Profile | Business Strategy |
        |---------|-----------------|-------------------|
        | **High-Value** | High spending, frequent purchases | VIP treatment, exclusive offers, retention focus |
        | **Growth Potential** | Medium spending, growing activity | Nurture with targeted promotions, loyalty programs |
        | **At-Risk** | Declining activity, low recent purchases | Win-back campaigns, special discounts |
        | **New/Occasional** | Few purchases, exploring | Welcome series, first-purchase incentives |

        **Revenue vs. Customer Distribution:**
        - If a small cluster contributes high revenue → These are your VIPs, protect them!
        - If a large cluster contributes low revenue → Opportunity to increase their spending

        **Action Items:**
        - Compare pie charts: Do customer % and revenue % match?
        - Big gap = opportunity for targeted marketing
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Cluster")
        chart_type = st.radio("Chart Type", ["Pie", "Bar"], horizontal=True, key="rev_chart")
        fig = create_revenue_by_cluster_chart(filtered_df, chart_type=chart_type.lower())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Customer Distribution by Cluster")
        fig = create_cluster_distribution_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster insights
    cluster_stats = filtered_df.groupby("Cluster").agg({
        "TotalPrice": "sum",
        "CustomerID": "nunique"
    }).reset_index()
    cluster_stats["Revenue_Pct"] = cluster_stats["TotalPrice"] / cluster_stats["TotalPrice"].sum() * 100
    cluster_stats["Customer_Pct"] = cluster_stats["CustomerID"] / cluster_stats["CustomerID"].sum() * 100
    cluster_stats["Value_Index"] = cluster_stats["Revenue_Pct"] / cluster_stats["Customer_Pct"]

    best_cluster = cluster_stats.loc[cluster_stats["Value_Index"].idxmax()]
    st.success(f"""
    🎯 **Key Insight:** Cluster {int(best_cluster['Cluster'])} has the highest value index
    ({best_cluster['Value_Index']:.2f}x) - {best_cluster['Revenue_Pct']:.1f}% of revenue from only
    {best_cluster['Customer_Pct']:.1f}% of customers. **Prioritize retention for this segment!**
    """)

    st.markdown("---")

    # ============ Purchase Patterns Section ============
    st.header("Purchase Patterns")

    with st.expander("ℹ️ **Reading the Heatmap** - When do customers buy?"):
        st.markdown("""
        The heatmap shows **purchase activity by day and hour**:

        - 🟡 **Bright/Yellow**: High activity periods - Peak shopping times
        - 🟣 **Dark/Purple**: Low activity periods - Quiet times

        **Business Applications:**

        | Pattern | Meaning | Action |
        |---------|---------|--------|
        | **Weekday peaks** | B2B customers or office workers | Target email campaigns for Tuesday-Thursday |
        | **Weekend peaks** | B2C/leisure shoppers | Social media ads for Saturday-Sunday |
        | **Morning peaks** | Early planners | Send promotions at 8-9 AM |
        | **Evening peaks** | After-work shoppers | Schedule flash sales for 6-8 PM |
        | **Lunch dips** | Break time | Avoid sending important emails at noon |

        **Pro Tip:** Schedule your marketing campaigns to hit **just before** peak times!
        """)

    fig = create_purchase_heatmap(filtered_df)
    st.plotly_chart(fig, use_container_width=True)

    # Find peak times
    heatmap_data = filtered_df.groupby(["DayOfWeek", "HourOfDay"]).size().reset_index(name="Count")
    peak = heatmap_data.loc[heatmap_data["Count"].idxmax()]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    st.info(f"""
    ⏰ **Peak Shopping Time:** {days[int(peak['DayOfWeek'])]} at {int(peak['HourOfDay']):02d}:00
    with {int(peak['Count']):,} transactions. **Schedule promotions for this time slot!**
    """)

    st.markdown("---")

    # ============ Product Analysis Section ============
    st.header("Product Analysis")

    with st.expander("ℹ️ **Understanding Product Performance** - What sells best?"):
        st.markdown("""
        Two views of product success:

        **By Quantity (Volume):**
        - Shows **most popular** items
        - High volume = customer favorites
        - Use for: Inventory planning, never run out of these!

        **By Revenue (Value):**
        - Shows **most profitable** items
        - High revenue = money makers
        - Use for: Promotion focus, premium positioning

        **Strategic Matrix:**

        | High Volume + High Revenue | High Volume + Low Revenue |
        |---------------------------|--------------------------|
        | ⭐ **Stars** - Promote heavily | 💰 **Cash Cows** - Consider price increase |

        | Low Volume + High Revenue | Low Volume + Low Revenue |
        |--------------------------|-------------------------|
        | 💎 **Hidden Gems** - Increase visibility | ❓ **Question Marks** - Review or discontinue |

        **Quick Wins:**
        - Bundle top quantity items with top revenue items
        - Create "Best Sellers" category on website
        - Never discount your stars unless strategic
        """)

    top_n = render_top_n_filter(default=10, max_val=30, key="products_n")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top {top_n} Products by Quantity")
        fig = create_top_products_chart(filtered_df, top_n=top_n, metric="quantity")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Top {top_n} Products by Revenue")
        fig = create_top_products_chart(filtered_df, top_n=top_n, metric="revenue")
        st.plotly_chart(fig, use_container_width=True)

    # Product insight
    top_qty = filtered_df.groupby("Description")["Quantity"].sum().nlargest(1)
    top_rev = filtered_df.groupby("Description")["TotalPrice"].sum().nlargest(1)

    if top_qty.index[0] == top_rev.index[0]:
        st.success(f"🌟 **Star Product:** '{top_qty.index[0]}' leads in BOTH quantity AND revenue. Ensure adequate stock!")
    else:
        st.info(f"""
        📊 **Product Mix Insight:**
        - Volume Leader: '{top_qty.index[0]}' (popular but may have lower margins)
        - Revenue Leader: '{top_rev.index[0]}' (high-value item)
        - **Strategy:** Bundle these together for maximum impact!
        """)

    st.markdown("---")

    # ============ Customer Analysis Section ============
    st.header("Customer Behavior Analysis")

    with st.expander("ℹ️ **Understanding Customer Distributions** - The Pareto Principle"):
        st.markdown("""
        These histograms reveal the **80/20 rule** (Pareto Principle) in action:

        **Transaction Distribution:**
        - Most customers make **few purchases** (left side of chart)
        - A small group makes **many purchases** (right tail)
        - The "long tail" customers are your **loyal base**

        **Spending Distribution:**
        - Most customers spend **small amounts**
        - Few customers spend **large amounts** (whales)
        - These "whales" often drive disproportionate revenue

        **What the Shape Tells You:**

        | Shape | Meaning | Strategy |
        |-------|---------|----------|
        | **Steep drop-off** | Many one-time buyers | Focus on second-purchase conversion |
        | **Fat tail** | Healthy repeat customer base | Nurture loyalty programs |
        | **Bimodal (two peaks)** | Two distinct customer groups | Create separate marketing tracks |

        **Key Actions:**
        - Move customers from "1 purchase" to "2+ purchases" (highest ROI)
        - Identify and protect your top 20% (they likely drive 80% revenue)
        - The mean line shows your "average" customer - everyone above is valuable!
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Distribution")
        fig = create_customer_distribution_chart(filtered_df, metric="transactions")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Spending Distribution")
        fig = create_customer_distribution_chart(filtered_df, metric="spending")
        st.plotly_chart(fig, use_container_width=True)

    # Pareto analysis
    customer_revenue = filtered_df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False)
    top_20_pct_customers = int(len(customer_revenue) * 0.2)
    top_20_revenue = customer_revenue.head(top_20_pct_customers).sum()
    pareto_ratio = top_20_revenue / customer_revenue.sum() * 100

    st.info(f"""
    📊 **Pareto Analysis:** Top 20% of customers ({top_20_pct_customers:,} customers)
    generate **{pareto_ratio:.1f}%** of total revenue.
    {"This follows the classic 80/20 rule!" if 75 < pareto_ratio < 85 else
     "Even more concentrated than typical - protect these VIPs!" if pareto_ratio > 85 else
     "More distributed than typical - good customer diversification."}
    """)

    st.markdown("---")

    # ============ Top Customers Table ============
    st.subheader("Top Customers by Revenue")

    with st.expander("ℹ️ **Using the Top Customers Table** - Identify VIPs"):
        st.markdown("""
        This table shows your most valuable customers. Use it to:

        1. **Identify VIPs** for personalized outreach
        2. **Spot concentration risk** if few customers dominate
        3. **Find patterns** in high-value customer behavior

        **Action Items:**
        - 📞 Personal account management for top 10
        - 🎁 Exclusive offers and early access
        - 🔔 Alert system if any VIP goes inactive
        - 📧 Thank you notes for milestone purchases
        """)

    top_n_customers = st.slider("Number of customers", 5, 50, 10, key="top_customers")
    render_top_customers_table(filtered_df, top_n=top_n_customers)

    st.markdown("---")

    # ============ RFM Analysis Section ============
    st.header("RFM Analysis")

    with st.expander("ℹ️ **Understanding RFM** - The Gold Standard of Customer Segmentation", expanded=True):
        st.markdown("""
        **RFM** stands for **Recency, Frequency, Monetary** - three key dimensions of customer value:

        | Metric | What It Measures | High Value Means | Low Value Means |
        |--------|------------------|------------------|-----------------|
        | **Recency** | Days since last purchase | Recent buyer (engaged) | Inactive (at risk) |
        | **Frequency** | Number of purchases | Loyal repeater | One-time buyer |
        | **Monetary** | Total amount spent | High spender (VIP) | Low spender |

        **RFM Segments & Strategies:**

        | Segment | R | F | M | Strategy |
        |---------|---|---|---|----------|
        | 🏆 **Champions** | Low | High | High | Reward, ask for referrals, early access |
        | 💎 **Loyal** | Low | High | Med | Upsell, loyalty program, VIP perks |
        | 🌟 **Potential Loyalist** | Low | Med | Med | Membership offers, personalized recommendations |
        | 🆕 **New Customers** | Low | Low | Low | Welcome series, onboarding, second-purchase incentive |
        | 😴 **At Risk** | High | High | High | Win-back campaigns, "We miss you" emails |
        | 💤 **Hibernating** | High | Low | Low | Aggressive discounts or let go |

        **Reading the Histograms:**
        - **Recency**: Left-skewed is good (most customers bought recently)
        - **Frequency**: Right-skewed is normal (few super-loyal customers)
        - **Monetary**: Right-skewed shows healthy "whale" customers
        """)

    # Filter RFM data by selected customers
    filtered_customer_ids = filtered_df["CustomerID"].unique()
    filtered_rfm = rfm[rfm["CustomerID"].isin(filtered_customer_ids)]

    if len(filtered_rfm) > 0:
        fig = create_rfm_distributions(filtered_rfm)
        st.plotly_chart(fig, use_container_width=True)

        # RFM-based insights
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📅 Recency Insight**")
            recent_30 = (filtered_rfm["Recency"] <= 30).sum()
            recent_pct = recent_30 / len(filtered_rfm) * 100
            st.metric("Active (≤30 days)", f"{recent_pct:.1f}%")
            if recent_pct > 50:
                st.success("Healthy engagement!")
            else:
                st.warning("Many inactive customers - run re-engagement campaign")

        with col2:
            st.markdown("**🔄 Frequency Insight**")
            repeat_rate = (filtered_rfm["Frequency"] > 1).sum() / len(filtered_rfm) * 100
            st.metric("Repeat Rate", f"{repeat_rate:.1f}%")
            if repeat_rate > 40:
                st.success("Strong repeat business!")
            else:
                st.warning("Focus on second-purchase conversion")

        with col3:
            st.markdown("**💰 Monetary Insight**")
            high_value = (filtered_rfm["Monetary"] > filtered_rfm["Monetary"].quantile(0.75)).sum()
            st.metric("High-Value Customers", f"{high_value:,}")
            st.info(f"Top 25% by spending")

        # RFM summary stats
        with st.expander("📊 View Detailed RFM Statistics"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Recency (Days)**")
                st.write(f"Mean: {filtered_rfm['Recency'].mean():.1f}")
                st.write(f"Median: {filtered_rfm['Recency'].median():.1f}")
                st.write(f"Min: {filtered_rfm['Recency'].min():.0f}")
                st.write(f"Max: {filtered_rfm['Recency'].max():.0f}")

            with col2:
                st.markdown("**Frequency (Transactions)**")
                st.write(f"Mean: {filtered_rfm['Frequency'].mean():.1f}")
                st.write(f"Median: {filtered_rfm['Frequency'].median():.1f}")
                st.write(f"Min: {filtered_rfm['Frequency'].min():.0f}")
                st.write(f"Max: {filtered_rfm['Frequency'].max():.0f}")

            with col3:
                st.markdown("**Monetary (£)**")
                st.write(f"Mean: £{filtered_rfm['Monetary'].mean():,.2f}")
                st.write(f"Median: £{filtered_rfm['Monetary'].median():,.2f}")
                st.write(f"Min: £{filtered_rfm['Monetary'].min():,.2f}")
                st.write(f"Max: £{filtered_rfm['Monetary'].max():,.2f}")
    else:
        st.warning("No RFM data available for selected filters.")


if __name__ == "__main__":
    main()
