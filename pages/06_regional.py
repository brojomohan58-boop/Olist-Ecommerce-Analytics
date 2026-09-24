"""
pages/06_regional.py
====================
Regional Analysis page.

Data grain
----------
- Customer-state metrics (orders, sales, delivery, reviews) use the
  order-level master (one row per order).
- Seller-state metrics (items, sales) use the item-level master.
- The two DataFrames are NEVER merged to avoid double-counting.

Layout
------
1. KPI scorecards — top states by orders / sales / delivery
2. Customer state: orders & sales
3. Customer state: delivery performance
4. Customer state: review scores
5. Seller state: items & sales
6. Data notes
"""

import streamlit as st

from src.data_loader import load_master, load_item_master
from src.filters import render_sidebar_filters, apply_master_filters, apply_item_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)

master_f = apply_master_filters(master, fs)
items_f  = apply_item_filters(items, fs)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🗺️ Regional Analysis")
st.caption(
    "Customer-state metrics use the order-level master (one row per order). "
    "Seller-state metrics use the item-level master. The two grains are never merged."
)

if len(master_f) == 0:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Pre-compute aggregates
# ---------------------------------------------------------------------------
state_sales   = kpis.sales_by_state(master_f)
state_del     = kpis.delivery_by_state(master_f)
state_reviews = kpis.review_by_state(master_f)
seller_state  = kpis.sales_by_seller_state(items_f)

# ---------------------------------------------------------------------------
# KPI scorecards
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

top_state_orders = state_sales.iloc[0]["customer_state"] if len(state_sales) > 0 else "N/A"
top_state_sales  = state_sales.iloc[0]["customer_state"] if len(state_sales) > 0 else "N/A"
n_states         = int(state_sales["customer_state"].nunique())
top_seller_state = seller_state.iloc[0]["seller_state"] if len(seller_state) > 0 else "N/A"
top_seller_state_pct = (
    seller_state.iloc[0]["total_sales_value"] / seller_state["total_sales_value"].sum() * 100
    if len(seller_state) > 0 else 0.0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(**charts.kpi_delta_card("Active States",          n_states,                fmt="{:,}"))
c2.metric(**charts.kpi_delta_card("Top State (Orders)",     top_state_orders,        fmt="{}"))
c3.metric(**charts.kpi_delta_card("Top Seller State",       top_seller_state,        fmt="{}"))
c4.metric(**charts.kpi_delta_card("Top Seller State Share", top_seller_state_pct,    fmt="{:.1f}%"))

# ---------------------------------------------------------------------------
# Customer state — orders & sales
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Orders & Sales by Customer State")

if len(state_sales) > 0:
    col_orders, col_sales = st.columns(2)

    with col_orders:
        fig_orders = charts.bar_horizontal(
            state_sales.head(15),
            x="order_count",
            y="customer_state",
            title="Orders by Customer State (Top 15)",
            x_axis_title="Orders",
        )
        st.plotly_chart(fig_orders, use_container_width=True)

    with col_sales:
        fig_sales = charts.bar_horizontal(
            state_sales.head(15),
            x="total_sales",
            y="customer_state",
            title="Sales Value by Customer State (Top 15)",
            x_axis_title="Sales (R$)",
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    with st.expander("Customer state — orders & sales table"):
        tbl = state_sales.copy()
        tbl["total_sales"]      = tbl["total_sales"].map("R$ {:,.0f}".format)
        tbl["avg_order_value"]  = tbl["avg_order_value"].map("R$ {:,.2f}".format)
        tbl["total_freight"]    = tbl["total_freight"].map("R$ {:,.0f}".format)
        tbl = tbl.rename(columns={
            "customer_state":    "State",
            "order_count":       "Orders",
            "total_sales":       "Total Sales",
            "avg_order_value":   "Avg Order Value",
            "total_freight":     "Total Freight",
            "unique_customers":  "Unique Customers",
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Customer state — delivery performance
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Delivery Performance by Customer State")
st.caption("Denominator: orders with known delivery time (null delivery_time_days excluded).")

if len(state_del) > 0:
    col_del, col_delay = st.columns(2)

    with col_del:
        fig_del = charts.bar_horizontal(
            state_del.head(15),
            x="avg_delivery_days",
            y="customer_state",
            title="Avg Delivery Days by State (Slowest 15)",
            x_axis_title="Avg Delivery Days",
        )
        st.plotly_chart(fig_del, use_container_width=True)

    with col_delay:
        state_del_sorted = state_del.sort_values("delay_rate", ascending=False)
        fig_delay = charts.bar_horizontal(
            state_del_sorted.head(15),
            x="delay_rate",
            y="customer_state",
            title="Delay Rate by State (Highest 15)",
            x_axis_title="Delay Rate",
        )
        st.plotly_chart(fig_delay, use_container_width=True)

    with st.expander("Customer state — delivery table"):
        tbl_del = state_del.copy()
        tbl_del["avg_delivery_days"]    = tbl_del["avg_delivery_days"].map("{:.1f}".format)
        tbl_del["median_delivery_days"] = tbl_del["median_delivery_days"].map("{:.1f}".format)
        tbl_del["delay_rate"]           = tbl_del["delay_rate"].map("{:.1%}".format)
        tbl_del = tbl_del.rename(columns={
            "customer_state":        "State",
            "order_count":           "Orders",
            "avg_delivery_days":     "Avg Delivery Days",
            "median_delivery_days":  "Median Delivery Days",
            "delay_rate":            "Delay Rate",
            "n_delayed":             "Delayed",
            "n_on_time":             "On Time",
        })
        st.dataframe(tbl_del, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Customer state — review scores
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Scores by Customer State")
st.caption("Sorted by average review score ascending (worst-rated states first).")

if len(state_reviews) > 0:
    col_rev, col_low = st.columns(2)

    with col_rev:
        fig_rev = charts.bar_horizontal(
            state_reviews.head(15),
            x="avg_review_score",
            y="customer_state",
            title="Avg Review Score by State (Lowest 15)",
            x_axis_title="Avg Score (1–5)",
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_low:
        state_rev_low = state_reviews.sort_values("low_score_rate", ascending=False)
        fig_low = charts.bar_horizontal(
            state_rev_low.head(15),
            x="low_score_rate",
            y="customer_state",
            title="1–2 Star Rate by State (Highest 15)",
            x_axis_title="1–2 Star Rate",
        )
        st.plotly_chart(fig_low, use_container_width=True)

    with st.expander("Customer state — review table"):
        tbl_rev = state_reviews.copy()
        tbl_rev["avg_review_score"] = tbl_rev["avg_review_score"].map("{:.3f}".format)
        tbl_rev["low_score_rate"]   = tbl_rev["low_score_rate"].map("{:.1%}".format)
        tbl_rev["five_star_rate"]   = tbl_rev["five_star_rate"].map("{:.1%}".format)
        tbl_rev = tbl_rev.rename(columns={
            "customer_state":    "State",
            "order_count":       "Orders",
            "avg_review_score":  "Avg Score",
            "n_reviews":         "Reviews",
            "low_score_rate":    "1–2 Star Rate",
            "five_star_rate":    "5-Star Rate",
        })
        st.dataframe(tbl_rev, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Seller state — items & sales
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Items & Sales by Seller State")
st.caption(
    "Uses item-level master (item grain). "
    "Revenue = sum of item prices. SP dominates due to marketplace concentration."
)

if len(seller_state) > 0:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig_ss_items = charts.bar_horizontal(
            seller_state.head(15),
            x="item_count",
            y="seller_state",
            title="Items Sold by Seller State (Top 15)",
            x_axis_title="Items",
        )
        st.plotly_chart(fig_ss_items, use_container_width=True)

    with col_s2:
        fig_ss_sales = charts.bar_horizontal(
            seller_state.head(15),
            x="total_sales_value",
            y="seller_state",
            title="Sales Value by Seller State (Top 15)",
            x_axis_title="Sales (R$)",
        )
        st.plotly_chart(fig_ss_sales, use_container_width=True)

    with st.expander("Seller state table"):
        tbl_ss = seller_state.copy()
        tbl_ss["total_sales_value"]    = tbl_ss["total_sales_value"].map("R$ {:,.0f}".format)
        tbl_ss["avg_item_price"]       = tbl_ss["avg_item_price"].map("R$ {:,.2f}".format)
        tbl_ss["avg_freight_per_item"] = tbl_ss["avg_freight_per_item"].map("R$ {:,.2f}".format)
        tbl_ss = tbl_ss.rename(columns={
            "seller_state":         "Seller State",
            "item_count":           "Items Sold",
            "unique_sellers":       "Unique Sellers",
            "unique_orders":        "Unique Orders",
            "total_sales_value":    "Total Sales",
            "avg_item_price":       "Avg Item Price",
            "avg_freight_per_item": "Avg Freight/Item",
        })
        st.dataframe(tbl_ss, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    n_orders = len(master_f)
    n_items  = len(items_f)
    st.markdown(
        f"""
        **Customer state** metrics are computed from the order-level master
        ({n_orders:,} orders in the selected period).

        **Seller state** metrics are computed from the item-level master
        ({n_items:,} items in the selected period).
        The two grains are **never merged** to avoid double-counting.

        **Delivery** denominator: orders with non-null `delivery_time_days`.
        **Review** denominator: orders with non-null `review_score`.

        **Active filters**: {fs.date_start} → {fs.date_end} ·
        customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}
        """
    )
