"""
pages/05_payments.py
====================
Payments page.

Data grain: order-level master only (one row per order).
No item-level data is used here.

Key rules
---------
- payment_type_share uses primary_payment_type (the dominant method per order).
- installment_distribution applies to credit-card orders only.
- aov_by_installment_band uses credit-card orders only.
- total_payment_value vs total_price difference is shown as a reconciliation
  metric only — no causal attribution is made.
"""

import streamlit as st

from src.data_loader import load_master, load_item_master
from src.filters import render_sidebar_filters, apply_master_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)

# All statuses for payment method share (payment exists even for cancelled orders)
fs_all  = FilterState(
    date_start=fs.date_start, date_end=fs.date_end,
    include_2016=fs.include_2016, order_statuses=[],
    customer_states=fs.customer_states,
)
master_all = apply_master_filters(master, fs_all)

# Delivered orders for AOV comparisons
master_f = apply_master_filters(master, fs)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("💳 Payments")
st.caption(
    "All payment metrics use the order-level master (one row per order). "
    "Payment-vs-sales difference is a reconciliation metric only — no causal attribution."
)

if len(master_all) == 0:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI scorecards
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

pts   = kpis.payment_type_share(master_all)
n_cc  = int((master_all["primary_payment_type"] == "credit_card").sum())
n_all = len(master_all)

cc_row   = pts[pts["primary_payment_type"] == "credit_card"]
cc_share = float(cc_row["share_pct"].values[0]) if len(cc_row) else 0.0
cc_aov   = float(cc_row["avg_order_value"].values[0]) if len(cc_row) else 0.0

instdist = kpis.installment_distribution(master_all)
avg_inst = (
    (instdist["max_installments"] * instdist["order_count"]).sum()
    / instdist["order_count"].sum()
    if len(instdist) > 0 else 0.0
)
high_inst = float(
    instdist.loc[instdist["max_installments"] > 6, "order_count"].sum()
    / instdist["order_count"].sum() * 100
    if len(instdist) > 0 else 0.0
)

aov_f = kpis.avg_order_value(master_f)

c1, c2, c3, c4 = st.columns(4)
c1.metric(**charts.kpi_delta_card("Credit Card Share",    cc_share,  fmt="{:.1f}%"))
c2.metric(**charts.kpi_delta_card("Avg Order Value",      aov_f,     fmt="R$ {:,.2f}"))
c3.metric(**charts.kpi_delta_card("Avg Installments (CC)", avg_inst, fmt="{:.2f}"))
c4.metric(**charts.kpi_delta_card("High-Installment (>6)", high_inst, fmt="{:.1f}%"))

st.caption(
    f"Payment type share: {n_all:,} orders (all statuses). "
    f"Installment KPIs: {n_cc:,} credit-card orders."
)

# ---------------------------------------------------------------------------
# Payment type share
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Payment Method Distribution")

col_donut, col_bar = st.columns([1, 1])
with col_donut:
    fig_donut = charts.donut_chart(
        labels=pts["primary_payment_type"].tolist(),
        values=pts["order_count"].tolist(),
        title="Payment Method Share",
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_bar:
    fig_aov_type = charts.bar_vertical(
        pts, x="primary_payment_type", y="avg_order_value",
        title="Avg Order Value by Payment Type",
        x_axis_title="Payment Type", y_axis_title="Avg Order Value (R$)",
    )
    st.plotly_chart(fig_aov_type, use_container_width=True)

# Payment method detail table
with st.expander("Payment type detail table"):
    tbl = pts.copy()
    tbl["share_pct"]        = tbl["share_pct"].map("{:.1f}%".format)
    tbl["avg_order_value"]  = tbl["avg_order_value"].map("R$ {:,.2f}".format)
    tbl["median_order_value"] = tbl["median_order_value"].map("R$ {:,.2f}".format)
    tbl = tbl.rename(columns={
        "primary_payment_type": "Payment Type", "order_count": "Orders",
        "share_pct": "Share", "avg_order_value": "Avg AOV",
        "median_order_value": "Median AOV",
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Instalment distribution (credit card only)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Instalment Distribution (Credit Card Orders)")
st.caption("Only orders with payment_type = credit_card are included.")

if len(instdist) > 0:
    fig_inst = charts.bar_vertical(
        instdist, x="max_installments", y="order_count",
        title="Instalment Count Distribution (Credit Card)",
        x_axis_title="Max Instalments", y_axis_title="Orders",
    )
    st.plotly_chart(fig_inst, use_container_width=True)
else:
    st.info("No credit-card orders in the selected period.")

# ---------------------------------------------------------------------------
# AOV by instalment band
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Average Order Value by Instalment Band (Credit Card)")
st.caption("Bands: 1 / 2–3 / 4–6 / 7–12 / 13+")

aib = kpis.aov_by_installment_band(master_all)
if len(aib) > 0:
    fig_aib = charts.bar_vertical(
        aib, x="band_label", y="avg_order_value",
        title="AOV by Instalment Band",
        x_axis_title="Instalment Band", y_axis_title="Avg Order Value (R$)",
    )
    st.plotly_chart(fig_aib, use_container_width=True)

# ---------------------------------------------------------------------------
# Payment-vs-sales reconciliation
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Payment vs Sales Reconciliation")
st.caption(
    "⚠️ The difference between `total_payment_value` and `total_price` is shown "
    "as a reconciliation metric only. No causal attribution is made."
)

pr = kpis.payment_reconciliation(master_all)
if len(pr) > 0:
    fig_recon = charts.grouped_bar(
        pr,
        x="primary_payment_type",
        y_cols=["avg_total_price", "avg_total_payment_value"],
        title="Avg Sales Value vs Payment Value by Payment Type",
        y_labels={"avg_total_price": "Avg Sales Value (R$)",
                   "avg_total_payment_value": "Avg Payment Value (R$)"},
        y_axis_title="R$",
        barmode="group",
    )
    st.plotly_chart(fig_recon, use_container_width=True)

    tbl_recon = pr.copy()
    tbl_recon["avg_total_price"]         = tbl_recon["avg_total_price"].map("R$ {:,.2f}".format)
    tbl_recon["avg_total_payment_value"] = tbl_recon["avg_total_payment_value"].map("R$ {:,.2f}".format)
    tbl_recon["avg_difference"]          = tbl_recon["avg_difference"].map("R$ {:+,.2f}".format)
    tbl_recon = tbl_recon.rename(columns={
        "primary_payment_type": "Payment Type", "order_count": "Orders",
        "avg_total_price": "Avg Sales",
        "avg_total_payment_value": "Avg Payment",
        "avg_difference": "Avg Difference",
    })
    st.dataframe(tbl_recon, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    st.markdown(
        f"""
        **Payment type share** uses all order statuses ({n_all:,} orders) in the period.
        `primary_payment_type` = the dominant payment method for each order.

        **Instalment KPIs** apply to credit-card orders only ({n_cc:,} orders).

        **Reconciliation**: `total_payment_value` may differ from `total_price` for various
        reasons not captured in this dataset. The difference is shown without attribution.

        **Active filters**: {fs.date_start} → {fs.date_end} ·
        customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}
        """
    )
