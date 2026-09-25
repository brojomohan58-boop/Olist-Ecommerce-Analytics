"""
pages/02_products_categories.py
================================
Products & Categories page.

Data grain
----------
- Sales / freight / item KPIs  : item-level master  (one row per order-item)
- Category delivery / reviews  : build_order_category_table() → one row per order
- No master (order-grain) data is used here — all figures come from the item grain.

Business questions answered
---------------------------
- Which categories generate the highest sales value and item volume?
- What is the typical item price and freight cost per category?
- How many sellers operate in each category?
- How do category sales trend over time?
"""

import streamlit as st

from src.data_loader import load_master, load_item_master, build_order_category_table
from src.filters import render_sidebar_filters, apply_item_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)
items_f = apply_item_filters(items, fs)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("📦 Products & Categories")
st.caption(
    "All figures use item-level data (one row per order-item). "
    "Sales value = SUM(item price). No order-level totals are used here."
)

if len(items_f) == 0:
    st.warning("No items match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Tier-1 KPI scorecards
# ---------------------------------------------------------------------------
sbc = kpis.sales_by_category(items_f)

total_items   = int(items_f["order_item_id"].count())
total_cat_sales = float(sbc["total_sales_value"].sum())
avg_price     = float(items_f["price"].mean())
avg_freight   = float(items_f["freight_value"].mean())
n_categories  = int(items_f["product_category_name_english"].nunique())
n_sellers     = int(items_f["seller_id"].nunique())

st.subheader("Key Performance Indicators")
c1, c2, c3 = st.columns(3)
c1.metric(**charts.kpi_delta_card("Items Sold",           total_items,     fmt="{:,.0f}"))
c2.metric(**charts.kpi_delta_card("Total Sales Value",    total_cat_sales, fmt="R$ {:,.0f}"))
c3.metric(**charts.kpi_delta_card("Avg Item Price",       avg_price,       fmt="R$ {:,.2f}"))

c4, c5, c6 = st.columns(3)
c4.metric(**charts.kpi_delta_card("Avg Freight / Item",   avg_freight,     fmt="R$ {:,.2f}"))
c5.metric(**charts.kpi_delta_card("Active Categories",    n_categories,    fmt="{:,.0f}"))
c6.metric(**charts.kpi_delta_card("Active Sellers",       n_sellers,       fmt="{:,.0f}"))

st.caption(
    "Sales Value = SUM(item price) from the item-level master. "
    "Unique orders: "
    f"{items_f['order_id'].nunique():,}."
)

# ---------------------------------------------------------------------------
# Top-N control
# ---------------------------------------------------------------------------
st.divider()
top_n = st.slider("Number of categories to display", min_value=5, max_value=72,
                  value=15, step=5, key="cat_top_n")

# ---------------------------------------------------------------------------
# Sales value by category
# ---------------------------------------------------------------------------
st.subheader("Sales Value by Category")
fig_sales = charts.bar_horizontal(
    sbc, x="total_sales_value", y="product_category_name_english",
    title=f"Top {top_n} Categories — Total Sales Value (R$)",
    top_n=top_n, x_axis_title="Sales Value (R$)",
)
st.plotly_chart(fig_sales, use_container_width=True)

# ---------------------------------------------------------------------------
# Items sold by category
# ---------------------------------------------------------------------------
st.subheader("Items Sold by Category")
fig_items = charts.bar_horizontal(
    sbc, x="item_count", y="product_category_name_english",
    title=f"Top {top_n} Categories — Items Sold",
    top_n=top_n, x_axis_title="Items Sold",
)
st.plotly_chart(fig_items, use_container_width=True)

# ---------------------------------------------------------------------------
# Avg item price by category
# ---------------------------------------------------------------------------
st.subheader("Average Item Price by Category")
fig_price = charts.bar_horizontal(
    sbc, x="avg_item_price", y="product_category_name_english",
    title=f"Top {top_n} Categories — Avg Item Price (R$)",
    top_n=top_n, x_axis_title="Avg Item Price (R$)",
)
st.plotly_chart(fig_price, use_container_width=True)

# ---------------------------------------------------------------------------
# Avg freight per item by category
# ---------------------------------------------------------------------------
st.subheader("Avg Freight per Item by Category")
fig_freight = charts.bar_horizontal(
    sbc, x="avg_freight_per_item", y="product_category_name_english",
    title=f"Top {top_n} Categories — Avg Freight per Item (R$)",
    top_n=top_n, x_axis_title="Avg Freight (R$)",
)
st.plotly_chart(fig_freight, use_container_width=True)

# ---------------------------------------------------------------------------
# Monthly item sales trend (filtered items)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Monthly Item Sales Trend")
ibm = kpis.items_by_month(items_f)
fig_trend = charts.line_trend(
    ibm, x="order_month_str",
    y_cols=["item_count", "total_item_sales_value"],
    title="Monthly Items Sold & Sales Value",
    y_labels={"item_count": "Items Sold",
               "total_item_sales_value": "Sales Value (R$)"},
    secondary_y_col="total_item_sales_value",
    y_axis_title="Items Sold",
    y2_axis_title="Sales Value (R$)",
)
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------------------------------------------------------
# Price vs volume scatter (category level)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Category Price vs Volume")
st.caption(
    "Bubble size = number of unique sellers in the category. "
    "Top categories by sales value are labelled."
)
fig_scatter = charts.scatter_plot(
    sbc,
    x="avg_item_price",
    y="item_count",
    title="Avg Item Price vs Items Sold (by Category)",
    size="unique_sellers",
    hover_name="product_category_name_english",
    hover_data=["total_sales_value", "unique_sellers"],
    x_axis_title="Avg Item Price (R$)",
    y_axis_title="Items Sold",
    max_points=len(sbc),   # all categories — cardinality is small (≤ 72)
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Category detail table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Category Detail Table")
display_sbc = sbc.rename(columns={
    "product_category_name_english": "Category",
    "item_count":        "Items Sold",
    "unique_orders":     "Unique Orders",
    "unique_sellers":    "Sellers",
    "total_sales_value": "Sales Value (R$)",
    "avg_item_price":    "Avg Price (R$)",
    "median_item_price": "Median Price (R$)",
    "avg_freight_per_item": "Avg Freight (R$)",
    "total_freight_value":  "Total Freight (R$)",
}).copy()
for col in ["Sales Value (R$)", "Avg Price (R$)", "Median Price (R$)",
            "Avg Freight (R$)", "Total Freight (R$)"]:
    display_sbc[col] = display_sbc[col].map("R$ {:,.2f}".format)

st.dataframe(display_sbc, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    st.markdown(
        f"""
        **Grain**: item-level master (one row per order-item).
        **Sales value** = `SUM(items.price)`, not `SUM(master.total_price)`.
        Category attribution is at item level — each item contributes its own price
        to its own category.

        **Active filters**: {fs.date_start} → {fs.date_end} · 
        statuses: {", ".join(fs.order_statuses) if fs.order_statuses else "all"} · 
        categories: {", ".join(fs.categories) if fs.categories else "all"} · 
        seller states: {", ".join(fs.seller_states) if fs.seller_states else "all"}

        **Rows in view**: {len(items_f):,} items · 
        {items_f['order_id'].nunique():,} unique orders · 
        {n_sellers:,} sellers · {n_categories:,} categories
        """
    )
