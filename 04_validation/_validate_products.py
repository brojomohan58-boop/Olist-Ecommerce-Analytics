"""_validate_products.py — headless validation of pages/02_products_categories.py."""
import sys, io, ast, time
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def passthrough_cache(*a, **k):
    if a and callable(a[0]): return a[0]
    def d(fn): return fn
    return d
st.cache_data = passthrough_cache

from src.data_loader import load_master, load_item_master, build_order_category_table
from src.filters import FilterState, apply_item_filters
import src.kpis as kpis
import src.charts as charts

passed = 0
failed = 0

def check(label, condition, detail=""):
    global passed, failed
    s = "PASS" if condition else "FAIL"
    print(f"  {s}  {label}" + (f"  [{detail}]" if detail else ""))
    if condition: passed += 1
    else:         failed += 1

def is_fig(obj): return isinstance(obj, go.Figure)

# ── 1. Syntax & wiring ────────────────────────────────────────────────────
print("=== 1. Syntax & app.py wiring ===")
for f in ["pages/02_products_categories.py", "app.py"]:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        check(f"Syntax OK: {f}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {f}", False, str(e))

app_src = open("app.py", encoding="utf-8").read()
check("02_products_categories.py in app.py", "pages/02_products_categories.py" in app_src)
check("page_products callable removed",      "def page_products" not in app_src)

# ── 2. Data loading ───────────────────────────────────────────────────────
print("\n=== 2. Data loading ===")
t0 = time.time()
master = load_master()
items  = load_item_master()
print(f"  Loaded in {time.time()-t0:.1f}s")
fs      = FilterState()
items_f = apply_item_filters(items, fs)
check("items_f non-empty", len(items_f) > 0, str(len(items_f)))

# ── 3. sales_by_category ─────────────────────────────────────────────────
print("\n=== 3. sales_by_category ===")
sbc = kpis.sales_by_category(items_f)
check("returns DataFrame",            isinstance(sbc, pd.DataFrame))
check("rows <= 72",                   len(sbc) <= 72, str(len(sbc)))
check("sorted by total_sales_value",  sbc["total_sales_value"].is_monotonic_decreasing)
check("all total_sales_value > 0",    (sbc["total_sales_value"] > 0).all())
check("item_count sum == len(items_f)",
      sbc["item_count"].sum() == len(items_f), str(sbc["item_count"].sum()))
check("required columns present",
      all(c in sbc.columns for c in [
          "product_category_name_english", "item_count", "unique_orders",
          "unique_sellers", "total_sales_value", "avg_item_price",
          "median_item_price", "avg_freight_per_item", "total_freight_value"]))
# grain guard: unique_orders per category never inflated by item count
check("unique_orders <= item_count per row",
      (sbc["unique_orders"] <= sbc["item_count"]).all())

# ── 4. Tier-1 KPI scalars ────────────────────────────────────────────────
print("\n=== 4. KPI scalars ===")
total_items     = int(items_f["order_item_id"].count())
total_cat_sales = float(sbc["total_sales_value"].sum())
avg_price       = float(items_f["price"].mean())
avg_freight     = float(items_f["freight_value"].mean())
n_categories    = int(items_f["product_category_name_english"].nunique())
n_sellers       = int(items_f["seller_id"].nunique())

check("total_items == len(items_f)",      total_items == len(items_f))
check("total_cat_sales > 0",              total_cat_sales > 0, f"R${total_cat_sales:,.0f}")
check("avg_price in range",               50 < avg_price < 200,  f"R${avg_price:.2f}")
check("avg_freight in range",             5  < avg_freight < 50, f"R${avg_freight:.2f}")
check("n_categories in [1,72]",           1 <= n_categories <= 72, str(n_categories))
check("n_sellers > 0",                    n_sellers > 0, str(n_sellers))

# ── 5. Grain separation ───────────────────────────────────────────────────
print("\n=== 5. Grain separation ===")
from src.filters import apply_master_filters
master_f = apply_master_filters(master, fs)
master_sales = float(master_f["total_price"].sum())
# Both should be numerically equal (items.price sums to same as order total_price)
check("item-grain total matches order-grain total",
      abs(total_cat_sales - master_sales) / max(master_sales, 1) < 0.01,
      f"item={total_cat_sales:,.0f}  order={master_sales:,.0f}")
# Confirm item-grain columns NOT on master, order-grain-only cols NOT on items
check("items_f has no total_price column",       "total_price"   not in items_f.columns)
check("items_f has no max_installments column",  "max_installments" not in items_f.columns)
check("items_f has price column",                "price"          in items_f.columns)
check("items_f has freight_value column",        "freight_value"  in items_f.columns)

# ── 6. items_by_month ─────────────────────────────────────────────────────
print("\n=== 6. items_by_month ===")
ibm = kpis.items_by_month(items_f)
check("returns DataFrame",          isinstance(ibm, pd.DataFrame))
check("sorted chronologically",     ibm["order_month_str"].is_monotonic_increasing)
check("item_count sum == len(items_f)",
      ibm["item_count"].sum() == len(items_f))
check("total_item_sales_value > 0", (ibm["total_item_sales_value"] > 0).all())

# ── 7. Chart outputs ──────────────────────────────────────────────────────
print("\n=== 7. Chart outputs ===")
for top_n in [5, 15, 72]:
    fig = charts.bar_horizontal(sbc, x="total_sales_value",
                                 y="product_category_name_english",
                                 title=f"Top {top_n} — Sales",
                                 top_n=top_n)
    actual_bars = len(fig.data[0].y)
    expected = min(top_n, len(sbc))
    check(f"bar_horizontal top_n={top_n}: bars={actual_bars}",
          actual_bars == expected, f"expected={expected}")

fig_items = charts.bar_horizontal(sbc, x="item_count",
                                   y="product_category_name_english",
                                   title="Items", top_n=15)
check("items bar returns Figure",  is_fig(fig_items))

fig_price = charts.bar_horizontal(sbc, x="avg_item_price",
                                   y="product_category_name_english",
                                   title="Avg Price", top_n=15)
check("price bar returns Figure",  is_fig(fig_price))

fig_freight = charts.bar_horizontal(sbc, x="avg_freight_per_item",
                                     y="product_category_name_english",
                                     title="Freight", top_n=15)
check("freight bar returns Figure", is_fig(fig_freight))

fig_trend = charts.line_trend(ibm, x="order_month_str",
                               y_cols=["item_count", "total_item_sales_value"],
                               title="Monthly trend",
                               secondary_y_col="total_item_sales_value",
                               y_axis_title="Items", y2_axis_title="Sales (R$)")
check("trend chart returns Figure",  is_fig(fig_trend))
check("trend chart has 2 traces",    len(fig_trend.data) == 2)
check("secondary y axis set",        fig_trend.data[1].yaxis == "y2")

fig_scatter = charts.scatter_plot(sbc, x="avg_item_price", y="item_count",
                                   title="Price vs Volume",
                                   size="unique_sellers",
                                   hover_name="product_category_name_english",
                                   max_points=len(sbc))
check("scatter returns Figure",      is_fig(fig_scatter))
check("scatter has traces",          len(fig_scatter.data) > 0)

# ── 8. kpi_delta_card formatting ─────────────────────────────────────────
print("\n=== 8. kpi_delta_card formatting ===")
for label, val, fmt in [
    ("Items Sold",         total_items,     "{:,.0f}"),
    ("Total Sales Value",  total_cat_sales, "R$ {:,.0f}"),
    ("Avg Item Price",     avg_price,       "R$ {:,.2f}"),
    ("Avg Freight / Item", avg_freight,     "R$ {:,.2f}"),
]:
    card = charts.kpi_delta_card(label, val, fmt=fmt)
    check(f"card '{label}' non-empty value", len(card["value"]) > 0, card["value"])

# ── 9. Edge case: category filter ─────────────────────────────────────────
print("\n=== 9. Edge case — single category filter ===")
fs_cat = FilterState(categories=["bed_bath_table"])
items_cat = apply_item_filters(items, fs_cat)
sbc_cat   = kpis.sales_by_category(items_cat)
check("single-category filter: 1 row",         len(sbc_cat) == 1, str(len(sbc_cat)))
check("single-category filter: correct cat",
      sbc_cat.iloc[0]["product_category_name_english"] == "bed_bath_table")
ibm_cat = kpis.items_by_month(items_cat)
check("items_by_month on single category works", isinstance(ibm_cat, pd.DataFrame))

# ── 10. Edge case: empty filter ───────────────────────────────────────────
print("\n=== 10. Edge case — empty filter ===")
fs_empty  = FilterState(categories=["NONEXISTENT_CATEGORY_XYZ"])
items_empty = apply_item_filters(items, fs_empty)
check("empty filter: 0 rows",   len(items_empty) == 0)
sbc_empty = kpis.sales_by_category(items_empty)
check("sales_by_category on empty returns empty DF",
      isinstance(sbc_empty, pd.DataFrame) and len(sbc_empty) == 0)
ibm_empty = kpis.items_by_month(items_empty)
check("items_by_month on empty returns empty DF",
      isinstance(ibm_empty, pd.DataFrame) and len(ibm_empty) == 0)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED")
    sys.exit(1)
