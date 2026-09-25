"""
pages/01_overview.py
====================
Executive Overview page — high-level KPI scorecards, monthly order &
revenue trends, and order status breakdown.

Data grain: order-level master only (one row per order).
No item-level data is used on this page.

Business questions answered
---------------------------
- How do orders and revenue change month-over-month?
- What is the on-time delivery rate and average review score?
- How are orders distributed across statuses?
- What is the average and median order value?
- How many unique customers placed orders in the selected period?
"""

import streamlit as st

from src.data_loader import load_master, load_item_master
from src.filters import render_sidebar_filters, apply_master_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load data (cached at app level; re-used across pages)
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
fs: FilterState = render_sidebar_filters(master, items)

# Filtered slices
# Default filter = delivered orders, Jan 2017 – Oct 2018, no 2016 partial.
master_f = apply_master_filters(master, fs)

# All-status slice for cancellation rate and status breakdown
fs_all = FilterState(
    date_start=fs.date_start,
    date_end=fs.date_end,
    include_2016=fs.include_2016,
    order_statuses=[],          # no status filter
    customer_states=fs.customer_states,
)
master_all = apply_master_filters(master, fs_all)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("📊 Executive Overview")
st.caption(
    "High-level KPIs, monthly order & revenue trends, and order status breakdown. "
    "Data grain: one row per order. Default view: delivered orders, Jan 2017 – Oct 2018."
)

if len(master_f) == 0:
    st.warning("No orders match the current filters. Adjust the sidebar selections.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI scorecards — row 1: volume & value
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

n_orders     = kpis.total_orders(master_f)
sales_val    = kpis.total_sales_value(master_f)
freight_val  = kpis.total_freight_value(master_f)
aov          = kpis.avg_order_value(master_f)
median_ov    = kpis.median_order_value(master_f)
n_customers  = kpis.unique_customers(master_f)
cancel_rate  = kpis.cancellation_rate(master_all)

col1, col2, col3, col4 = st.columns(4)
col1.metric(**charts.kpi_delta_card("Total Orders",       n_orders,    fmt="{:,.0f}"))
col2.metric(**charts.kpi_delta_card("Total Sales Value",  sales_val,   fmt="R$ {:,.0f}"))
col3.metric(**charts.kpi_delta_card("Total Freight",      freight_val, fmt="R$ {:,.0f}"))
col4.metric(**charts.kpi_delta_card("Avg Order Value",    aov,         fmt="R$ {:,.2f}"))

col5, col6, col7, col8 = st.columns(4)
col5.metric(**charts.kpi_delta_card("Median Order Value", median_ov,   fmt="R$ {:,.2f}"))
col6.metric(**charts.kpi_delta_card("Unique Customers",   n_customers, fmt="{:,.0f}"))
col7.metric(**charts.kpi_delta_card("Cancellation Rate",  cancel_rate * 100,
                                    fmt="{:.2f}%",
                                    delta_color="inverse"))

# ---------------------------------------------------------------------------
# KPI scorecards — row 2: delivery & reviews (delivered orders only)
# ---------------------------------------------------------------------------
st.divider()

otr         = kpis.on_time_rate(master_f)
dr          = kpis.delay_rate(master_f)
avg_del     = kpis.avg_delivery_days(master_f)
med_del     = kpis.median_delivery_days(master_f)
avg_late    = kpis.avg_late_delay_days(master_f)
avg_review  = kpis.avg_review_score(master_f)
five_star   = kpis.five_star_rate(master_f)
low_score   = kpis.low_score_rate(master_f)
rev_cov     = kpis.review_coverage(master_f)

col9, col10, col11, col12 = st.columns(4)
col9.metric(**charts.kpi_delta_card("On-Time Delivery",   otr * 100,   fmt="{:.1f}%"))
col10.metric(**charts.kpi_delta_card("Delay Rate",        dr * 100,    fmt="{:.1f}%",
                                     delta_color="inverse"))
col11.metric(**charts.kpi_delta_card("Avg Delivery Days", avg_del,     fmt="{:.1f} days"))
col12.metric(**charts.kpi_delta_card("Median Delivery",   med_del,     fmt="{:.1f} days"))

col13, col14, col15, col16 = st.columns(4)
col13.metric(**charts.kpi_delta_card("Avg Delay (late)",  avg_late,    fmt="{:.1f} days",
                                     delta_color="inverse"))
col14.metric(**charts.kpi_delta_card("Avg Review Score",  avg_review,  fmt="{:.2f} / 5"))
col15.metric(**charts.kpi_delta_card("5-Star Rate",       five_star * 100, fmt="{:.1f}%"))
col16.metric(**charts.kpi_delta_card("1–2 Star Rate",     low_score * 100, fmt="{:.1f}%",
                                     delta_color="inverse"))

# Delivery denominator note
n_with_delivery = int(master_f["delivery_time_days"].notna().sum())
n_with_review   = int(master_f["review_score"].notna().sum())
st.caption(
    f"Delivery KPIs: {n_with_delivery:,} orders with resolved delivery date. "
    f"Review KPIs: {n_with_review:,} orders with a review score "
    f"({rev_cov * 100:.1f}% coverage). "
    "Avg Delay shows late orders (is_delayed = 1) only."
)

# ---------------------------------------------------------------------------
# Monthly trend — orders & sales (dual-axis)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Monthly Order Volume & Sales Value")

monthly = kpis.monthly_orders_and_sales(master_f)

if len(monthly) > 0:
    # 2016 partial-period annotation flag
    has_2016 = fs.include_2016 and any(
        str(m).startswith("2016") for m in monthly["order_month_str"]
    )

    fig_trend = charts.line_trend(
        monthly,
        x="order_month_str",
        y_cols=["order_count", "total_sales"],
        title="Monthly Orders & Sales Value",
        y_labels={"order_count": "Orders", "total_sales": "Sales Value (R$)"},
        secondary_y_col="total_sales",
        y_axis_title="Orders",
        y2_axis_title="Sales Value (R$)",
    )

    if has_2016:
        fig_trend.add_annotation(
            text="◀ 2016 partial period",
            x=monthly["order_month_str"].iloc[0],
            y=monthly["order_count"].iloc[0],
            showarrow=True, arrowhead=2,
            font=dict(size=11, color="#57606a"),
            ax=60, ay=-30,
        )

    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No monthly data available for the selected period.")

# ---------------------------------------------------------------------------
# Monthly AOV trend
# ---------------------------------------------------------------------------
st.subheader("Monthly Average Order Value")

fig_aov = charts.line_trend(
    monthly,
    x="order_month_str",
    y_cols=["avg_order_value"],
    title="Monthly Average Order Value (R$)",
    y_labels={"avg_order_value": "AOV (R$)"},
    y_axis_title="R$",
)
st.plotly_chart(fig_aov, use_container_width=True)

# ---------------------------------------------------------------------------
# Order status breakdown
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Order Status Distribution")
st.caption(
    "Shows all order statuses in the selected date range, regardless of the "
    "status filter applied to delivery and value KPIs above."
)

obs = kpis.orders_by_status(master_all)

col_donut, col_table = st.columns([1, 1])
with col_donut:
    fig_status = charts.donut_chart(
        labels=obs["order_status"].tolist(),
        values=obs["order_count"].tolist(),
        title="Order Status Share",
    )
    st.plotly_chart(fig_status, use_container_width=True)

with col_table:
    st.markdown("**Order counts by status**")
    display_obs = obs.copy()
    display_obs["share_pct"] = display_obs["share_pct"].map("{:.2f}%".format)
    display_obs.columns = ["Status", "Orders", "Share"]
    st.dataframe(display_obs, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Dataset / filter transparency footer
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes & filter summary", expanded=False):
    st.markdown(
        f"""
        **Active filters**
        - Date range: {fs.date_start} → {fs.date_end}
        - 2016 partial period: {"included" if fs.include_2016 else "excluded"}
        - Order statuses (KPI rows): {", ".join(fs.order_statuses) if fs.order_statuses else "all"}
        - Customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}

        **Denominator rules**
        - Total Orders, Sales, Customers: all rows matching the status filter.
        - Delivery KPIs: orders with a non-null `delivery_time_days` ({n_with_delivery:,} rows).
        - Delay KPIs: orders with a non-null `is_delayed` flag.
        - Avg Delay (late): orders where `is_delayed == 1` ({int((master_f["is_delayed"] == 1).sum()):,} rows).
        - Review KPIs: orders with a non-null `review_score` ({n_with_review:,} rows).
        - Cancellation Rate: {int((master_all["order_status"] == "canceled").sum()):,} cancelled /
          {len(master_all):,} total orders in period.

        **Data grain**: this page uses the order-level master only (one row per order).
        No item-level data is used. No double-counting of orders, payments, or reviews.
        """
    )
