"""
pages/07_sellers.py
===================
Seller Intelligence page.

Data grain
----------
- Seller performance metrics use the item-level master.
- One row represents one order item.
- Order-level and item-level data are not merged here to avoid double-counting.

Planned analysis
----------------
1. Seller KPI scorecards
2. Top sellers by sales value
3. Top sellers by item/order activity
4. Seller delivery performance
5. Seller freight performance
6. Seller performance comparison
7. Data notes
"""

import streamlit as st

from src.data_loader import load_master, load_item_master
from src.filters import (
    render_sidebar_filters,
    apply_master_filters,
    apply_item_filters,
    FilterState,
)
import src.kpis as kpis
import src.charts as charts


# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)

master_f = apply_master_filters(master, fs)
items_f = apply_item_filters(items, fs)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🏪 Seller Intelligence")
st.caption(
    "Seller performance analysis using the item-level dataset. "
    "Sales, product activity, freight, and operational metrics are evaluated "
    "without permanently merging the order and item grains."
)

if len(items_f) == 0:
    st.warning("No seller records match the current filters.")
    st.stop()


# ---------------------------------------------------------------------------
# Seller performance aggregates
# ---------------------------------------------------------------------------

seller_sales = (
    items_f.groupby("seller_id", as_index=False)
    .agg(
        total_sales_value=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        items_sold=("order_item_id", "count"),
        unique_orders=("order_id", "nunique"),
    )
)

seller_sales["avg_item_price"] = (
    seller_sales["total_sales_value"] / seller_sales["items_sold"]
)

seller_sales["freight_per_item"] = (
    seller_sales["total_freight_value"] / seller_sales["items_sold"]
)


# ---------------------------------------------------------------------------
# Seller KPI scorecards
# ---------------------------------------------------------------------------

st.subheader("Key Performance Indicators")

n_sellers = int(seller_sales["seller_id"].nunique())

total_sales = float(seller_sales["total_sales_value"].sum())

total_items = int(seller_sales["items_sold"].sum())

avg_sales_per_seller = (
    total_sales / n_sellers
    if n_sellers > 0
    else 0.0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Active Sellers",
    f"{n_sellers:,}",
)

c2.metric(
    "Seller Sales Value",
    f"R$ {total_sales:,.0f}",
)

c3.metric(
    "Items Sold",
    f"{total_items:,}",
)

c4.metric(
    "Avg Sales / Seller",
    f"R$ {avg_sales_per_seller:,.0f}",
)

# ---------------------------------------------------------------------------
# Top sellers by sales value
# ---------------------------------------------------------------------------

st.subheader("Top Sellers by Sales Value")

top_sellers_sales = seller_sales.sort_values(
    "total_sales_value",
    ascending=False,
).head(15)

fig_sales = charts.bar_horizontal(
    top_sellers_sales,
    x="total_sales_value",
    y="seller_id",
    title="Top 15 Sellers by Sales Value",
    x_axis_title="Sales Value (R$)",
)
st.plotly_chart(fig_sales, use_container_width=True)

# ---------------------------------------------------------------------------
# Top sellers by item and order activity
# ---------------------------------------------------------------------------

st.subheader("Top Sellers by Activity")

top_sellers_activity = seller_sales.sort_values(
    "items_sold",
    ascending=False,
).head(15)

fig_activity = charts.bar_horizontal(
    top_sellers_activity,
    x="items_sold",
    y="seller_id",
    title="Top 15 Sellers by Items Sold",
    x_axis_title="Items Sold",
)

st.plotly_chart(fig_activity, use_container_width=True)


# ---------------------------------------------------------------------------
# Seller freight performance
# ---------------------------------------------------------------------------

st.subheader("Seller Freight Performance")

top_sellers_freight = seller_sales.sort_values(
    "freight_per_item",
    ascending=False,
).head(15)

fig_freight = charts.bar_horizontal(
    top_sellers_freight,
    x="freight_per_item",
    y="seller_id",
    title="Top 15 Sellers by Freight per Item",
    x_axis_title="Freight per Item (R$)",
)

st.plotly_chart(fig_freight, use_container_width=True)

st.info(
    f"Seller Intelligence dataset ready: "
    f"{len(items_f):,} filtered order-item records."
)