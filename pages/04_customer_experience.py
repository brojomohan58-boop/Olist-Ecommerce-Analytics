"""
pages/04_customer_experience.py
================================
Customer Experience & Reviews page.

Data grain
----------
- Review score KPIs, distribution, monthly trend,
  review vs delay, state-level reviews : order-level master (one row per order)
- Category-level reviews               : build_order_category_table()
  → one row per order, primary_category assigned by mode with alpha tie-break

Key rules
---------
- Review KPIs exclude rows with null review_score; denominator shown in UI.
- Review coverage = reviews with score / total orders.
- Association between delay and review is shown without causal attribution.
- 5-star rate and 1-2 star rate use non-null review_score as denominator.
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

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("⭐ Customer Experience")
st.caption(
    "Review KPIs use the order-level master (one row per order). "
    "Category review scores use an order-level attribution table — each order counted once."
)

if len(master_f) == 0:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI scorecards
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

avg_rev  = kpis.avg_review_score(master_f)
five_s   = kpis.five_star_rate(master_f)
low_s    = kpis.low_score_rate(master_f)
rev_cov  = kpis.review_coverage(master_f)
n_reviews = int(master_f["review_score"].notna().sum())
n_orders  = kpis.total_orders(master_f)

c1, c2, c3, c4 = st.columns(4)
c1.metric(**charts.kpi_delta_card("Avg Review Score",  avg_rev,       fmt="{:.2f} / 5"))
c2.metric(**charts.kpi_delta_card("5-Star Rate",       five_s * 100,  fmt="{:.1f}%"))
c3.metric(**charts.kpi_delta_card("1–2 Star Rate",     low_s * 100,   fmt="{:.1f}%",
                                   delta_color="inverse"))
c4.metric(**charts.kpi_delta_card("Review Coverage",   rev_cov * 100, fmt="{:.1f}%"))

st.caption(
    f"Review KPIs: {n_reviews:,} orders with a review score "
    f"({rev_cov*100:.1f}% of {n_orders:,} orders in period)."
)

# ---------------------------------------------------------------------------
# Review score distribution
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Score Distribution")

rsd = kpis.review_score_distribution(master_f)
rsd["label"] = rsd["review_score"].astype(str) + " ★  " + rsd["share_pct"].map("({:.1f}%)".format)

fig_dist = charts.bar_vertical(
    rsd, x="review_score", y="count",
    title="Review Score Distribution",
    x_axis_title="Review Score", y_axis_title="Orders",
    text_col="label",
    color_discrete_sequence=["#3b82d4"],
)
st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------------------------
# Monthly avg review trend
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Monthly Average Review Score")

monthly_rev = (
    master_f.dropna(subset=["review_score"])
    .groupby("order_month_str", sort=True)
    .agg(avg_review_score=("review_score", "mean"),
         n_reviews=("review_score", "count"))
    .reset_index()
)

if len(monthly_rev) > 0:
    fig_rev_trend = charts.line_trend(
        monthly_rev,
        x="order_month_str",
        y_cols=["avg_review_score"],
        title="Monthly Avg Review Score",
        y_labels={"avg_review_score": "Avg Review Score"},
        y_axis_title="Avg Score",
    )
    st.plotly_chart(fig_rev_trend, use_container_width=True)

# ---------------------------------------------------------------------------
# Review score vs delivery delay (association — no causation claim)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Score: On-Time vs Delayed Orders")
st.caption(
    "⚠️ This chart shows an **association** between delivery status and review scores. "
    "It does not establish a causal relationship."
)

rvd = kpis.review_vs_delay(master_f)

if len(rvd) == 2:
    rvd_display = rvd.copy()
    rvd_display["delivery_status"] = rvd_display["is_delayed"].map(
        {0: "On-Time", 1: "Delayed"}
    ).fillna(rvd_display["is_delayed"].astype(str))

    col_bar, col_table = st.columns([2, 1])
    with col_bar:
        fig_rvd = charts.grouped_bar(
            rvd_display,
            x="delivery_status",
            y_cols=["avg_review_score"],
            title="Avg Review Score by Delivery Status",
            y_labels={"avg_review_score": "Avg Review Score"},
            y_axis_title="Avg Review Score",
        )
        st.plotly_chart(fig_rvd, use_container_width=True)

    with col_table:
        st.markdown("**Summary**")
        tbl = rvd_display[["delivery_status", "order_count",
                            "avg_review_score", "low_score_rate", "five_star_rate"]].copy()
        tbl["avg_review_score"] = tbl["avg_review_score"].map("{:.2f}".format)
        tbl["low_score_rate"]   = tbl["low_score_rate"].map("{:.1%}".format)
        tbl["five_star_rate"]   = tbl["five_star_rate"].map("{:.1%}".format)
        tbl = tbl.rename(columns={
            "delivery_status": "Status", "order_count": "Orders",
            "avg_review_score": "Avg Score", "low_score_rate": "1-2★ Rate",
            "five_star_rate": "5★ Rate",
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)
else:
    st.info("Delay data not available for the current filter selection.")

# ---------------------------------------------------------------------------
# Review score by customer state
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Score by Customer State")

rbs = kpis.review_by_state(master_f)

tab_avg, tab_low = st.tabs(["Avg Review Score", "1–2 Star Rate"])
with tab_avg:
    fig_state_rev = charts.bar_horizontal(
        rbs, x="avg_review_score", y="customer_state",
        title="Avg Review Score by Customer State (worst first)",
        top_n=27, x_axis_title="Avg Review Score",
    )
    st.plotly_chart(fig_state_rev, use_container_width=True)
with tab_low:
    rbs_pct = rbs.copy()
    rbs_pct["low_score_pct"] = rbs_pct["low_score_rate"] * 100
    fig_state_low = charts.bar_horizontal(
        rbs_pct, x="low_score_pct", y="customer_state",
        title="1–2 Star Rate (%) by Customer State",
        top_n=27, x_axis_title="1–2 Star Rate (%)",
    )
    st.plotly_chart(fig_state_low, use_container_width=True)

# ---------------------------------------------------------------------------
# Category-level review scores  (order-category table)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Score by Category")
st.caption(
    "Each order is counted once, attributed to its most frequently purchased category. "
    "Multi-category orders attributed to dominant category."
)

oct_df = build_order_category_table(items_f)
rbc    = kpis.review_by_category(oct_df)

top_n_cat = st.slider("Categories to display", 5, min(72, len(rbc)), 20,
                       step=5, key="rev_cat_top_n")

tab_cat_avg, tab_cat_low = st.tabs(["Avg Review Score", "1–2 Star Rate"])
with tab_cat_avg:
    fig_cat_avg = charts.bar_horizontal(
        rbc, x="avg_review_score", y="primary_category",
        title=f"Bottom {top_n_cat} Categories — Avg Review Score",
        top_n=top_n_cat, x_axis_title="Avg Review Score",
        add_multicategory_note=True,
    )
    st.plotly_chart(fig_cat_avg, use_container_width=True)
with tab_cat_low:
    rbc_pct = rbc.copy()
    rbc_pct["low_score_pct"] = rbc_pct["low_score_rate"] * 100
    fig_cat_low = charts.bar_horizontal(
        rbc_pct, x="low_score_pct", y="primary_category",
        title=f"Top {top_n_cat} Categories — 1–2 Star Rate (%)",
        top_n=top_n_cat, x_axis_title="1–2 Star Rate (%)",
        add_multicategory_note=True,
    )
    st.plotly_chart(fig_cat_low, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    st.markdown(
        f"""
        **Review KPI denominator**: {n_reviews:,} orders with a non-null `review_score`
        out of {n_orders:,} total orders ({rev_cov*100:.1f}% coverage).

        **Delay association**: Orders with delayed delivery show a different review score
        distribution. This is an **association**, not a proven causal relationship —
        other factors may also influence review scores.

        **Category reviews**: built from `build_order_category_table()` — one row per order.
        {int(oct_df["has_multiple_categories"].sum()):,} orders attributed to dominant category.

        **Active filters**: {fs.date_start} → {fs.date_end} ·
        statuses: {", ".join(fs.order_statuses) if fs.order_statuses else "all"} ·
        customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}
        """
    )
