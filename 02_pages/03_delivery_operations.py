"""
pages/03_delivery_operations.py
================================
Delivery & Operations page.

Data grain
----------
- Overall KPIs, state analysis, monthly trend : order-level master (delivered)
- Category-level delivery                     : build_order_category_table()
  → one row per order; category assigned via mode with alpha tie-break.

Key rules
---------
- All delivery KPIs exclude rows where delivery_time_days is null.
- delay_days sign: negative = arrived early.
  "Avg Delay" KPI uses only is_delayed == 1 rows.
- Category delivery denominator = unique orders (not item rows).
- Multi-category orders attributed to primary category (documented in UI).
"""

import streamlit as st

from src.data_loader import load_master, load_item_master, build_order_category_table
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

# For cancellation rate: all statuses
fs_all   = FilterState(
    date_start=fs.date_start, date_end=fs.date_end,
    include_2016=fs.include_2016, order_statuses=[],
    customer_states=fs.customer_states,
)
master_all = apply_master_filters(master, fs_all)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🚚 Delivery & Operations")
st.caption(
    "Overall KPIs and state-level analysis use the order-level master (one row per order). "
    "Category-level delivery uses an order-level category attribution table — each order "
    "counted once."
)

if len(master_f) == 0:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Overall KPI scorecards
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

avg_del   = kpis.avg_delivery_days(master_f)
med_del   = kpis.median_delivery_days(master_f)
otr       = kpis.on_time_rate(master_f)
dr        = kpis.delay_rate(master_f)
avg_late  = kpis.avg_late_delay_days(master_f)
cancel_r  = kpis.cancellation_rate(master_all)

n_del     = int(master_f["delivery_time_days"].notna().sum())
n_delayed = int((master_f["is_delayed"] == 1).sum())

c1, c2, c3 = st.columns(3)
c1.metric(**charts.kpi_delta_card("Avg Delivery Time",   avg_del,          fmt="{:.1f} days"))
c2.metric(**charts.kpi_delta_card("Median Delivery Time", med_del,         fmt="{:.1f} days"))
c3.metric(**charts.kpi_delta_card("On-Time Delivery Rate", otr * 100,      fmt="{:.1f}%"))

c4, c5, c6 = st.columns(3)
c4.metric(**charts.kpi_delta_card("Delay Rate",          dr * 100,         fmt="{:.1f}%",
                                   delta_color="inverse"))
c5.metric(**charts.kpi_delta_card("Avg Delay (late orders)", avg_late,     fmt="{:.1f} days",
                                   delta_color="inverse"))
c6.metric(**charts.kpi_delta_card("Cancellation Rate",   cancel_r * 100,   fmt="{:.2f}%",
                                   delta_color="inverse"))

st.caption(
    f"Delivery KPIs: {n_del:,} orders with resolved delivery date. "
    f"Avg Delay: {n_delayed:,} late orders (is_delayed = 1). "
    "Negative delay_days = arrived early — excluded from Avg Delay."
)

# ---------------------------------------------------------------------------
# Delivery time distribution
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Delivery Time Distribution")

fig_hist = charts.histogram(
    master_f["delivery_time_days"].dropna(),
    title="Delivery Time Distribution",
    xaxis_title="Days from purchase to delivery",
    nbins=40,
    clip_upper=60,
    add_delay_note=False,
)
st.plotly_chart(fig_hist, use_container_width=True)
st.caption("Values above 60 days clipped for readability. See table below for full range.")

# On-time vs late donut
col_donut, col_stats = st.columns([1, 1])
with col_donut:
    fig_donut = charts.donut_chart(
        labels=["On-Time", "Late"],
        values=[int((master_f["is_delayed"] == 0).sum()),
                int((master_f["is_delayed"] == 1).sum())],
        title="On-Time vs Late Orders",
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_stats:
    st.markdown("**Delivery time summary**")
    del_series = master_f["delivery_time_days"].dropna()
    st.dataframe(
        del_series.describe().rename({
            "count": "Orders", "mean": "Mean (days)", "std": "Std Dev",
            "min": "Min", "25%": "25th pct", "50%": "Median",
            "75%": "75th pct", "max": "Max",
        }).to_frame("Value").reset_index().rename(columns={"index": "Metric"}),
        hide_index=True, use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Monthly delivery trend
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Monthly Delivery Performance")

dbm = kpis.delivery_by_month(master_f)

fig_monthly = charts.line_trend(
    dbm,
    x="order_month_str",
    y_cols=["avg_delivery_days", "delay_rate"],
    title="Monthly Avg Delivery Days & Delay Rate",
    y_labels={"avg_delivery_days": "Avg Delivery (days)",
               "delay_rate": "Delay Rate"},
    secondary_y_col="delay_rate",
    y_axis_title="Avg Delivery (days)",
    y2_axis_title="Delay Rate",
    add_delay_note=True,
)
st.plotly_chart(fig_monthly, use_container_width=True)

# ---------------------------------------------------------------------------
# Delivery by customer state
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Delivery Performance by Customer State")

dbs = kpis.delivery_by_state(master_f)

tab_del, tab_delay = st.tabs(["Avg Delivery Days", "Delay Rate"])
with tab_del:
    fig_state_del = charts.bar_horizontal(
        dbs, x="avg_delivery_days", y="customer_state",
        title="Avg Delivery Days by Customer State",
        top_n=27, x_axis_title="Avg Delivery Days",
    )
    st.plotly_chart(fig_state_del, use_container_width=True)

with tab_delay:
    # Convert delay_rate to percentage for display
    dbs_pct = dbs.copy()
    dbs_pct["delay_rate_pct"] = dbs_pct["delay_rate"] * 100
    fig_state_dr = charts.bar_horizontal(
        dbs_pct, x="delay_rate_pct", y="customer_state",
        title="Delay Rate (%) by Customer State",
        top_n=27, x_axis_title="Delay Rate (%)",
        add_multicategory_note=False,
    )
    st.plotly_chart(fig_state_dr, use_container_width=True)

with st.expander("State delivery detail table"):
    display_dbs = dbs.copy()
    display_dbs["delay_rate"] = display_dbs["delay_rate"].map("{:.1%}".format)
    display_dbs["avg_delivery_days"]    = display_dbs["avg_delivery_days"].map("{:.1f}".format)
    display_dbs["median_delivery_days"] = display_dbs["median_delivery_days"].map("{:.1f}".format)
    display_dbs = display_dbs.rename(columns={
        "customer_state": "State", "order_count": "Orders",
        "avg_delivery_days": "Avg Days", "median_delivery_days": "Median Days",
        "delay_rate": "Delay Rate", "n_delayed": "Late", "n_on_time": "On-Time",
    })
    st.dataframe(display_dbs, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Category-level delivery  (order-category table — one row per order)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Delivery Performance by Category")
st.caption(
    "Each order is counted once, attributed to its most frequently purchased category. "
    "Orders with items from multiple categories are attributed to the dominant category."
)

oct_df = build_order_category_table(items_f)
dbc    = kpis.delivery_by_category(oct_df)

top_n_cat = st.slider("Categories to display", 5, min(72, len(dbc)), 20,
                       step=5, key="del_cat_top_n")

tab_cat_del, tab_cat_dr = st.tabs(["Avg Delivery Days", "Delay Rate"])
with tab_cat_del:
    fig_cat_del = charts.bar_horizontal(
        dbc, x="avg_delivery_days", y="primary_category",
        title=f"Top {top_n_cat} Categories — Avg Delivery Days",
        top_n=top_n_cat, x_axis_title="Avg Delivery Days",
        add_multicategory_note=True,
    )
    st.plotly_chart(fig_cat_del, use_container_width=True)

with tab_cat_dr:
    dbc_pct = dbc.copy()
    dbc_pct["delay_rate_pct"] = dbc_pct["delay_rate"] * 100
    fig_cat_dr = charts.bar_horizontal(
        dbc_pct, x="delay_rate_pct", y="primary_category",
        title=f"Top {top_n_cat} Categories — Delay Rate (%)",
        top_n=top_n_cat, x_axis_title="Delay Rate (%)",
        add_multicategory_note=True,
    )
    st.plotly_chart(fig_cat_dr, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    st.markdown(
        f"""
        **Delivery KPI denominator**: {n_del:,} orders with a resolved
        `delivery_time_days` (orders without a delivery date are excluded).

        **Delay sign convention**: `delay_days` is negative when an order
        arrives *before* the estimated date. The Avg Delay KPI uses only
        orders flagged `is_delayed = 1` ({n_delayed:,} orders).

        **Category delivery**: built from `build_order_category_table()` —
        one row per order. {int(oct_df['has_multiple_categories'].sum()):,} orders
        contained items from multiple categories and were attributed to the
        most frequently purchased category.

        **Active filters**: {fs.date_start} → {fs.date_end} ·
        statuses: {", ".join(fs.order_statuses) if fs.order_statuses else "all"} ·
        customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}
        """
    )
