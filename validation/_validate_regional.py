"""_validate_regional.py — headless validation of pages/06_regional.py."""
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

from src.data_loader import load_master, load_item_master
from src.filters import FilterState, apply_master_filters, apply_item_filters
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
for f in ["pages/06_regional.py", "app.py"]:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        check(f"Syntax OK: {f}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {f}", False, str(e))

app_src = open("app.py", encoding="utf-8").read()
check("06_regional.py in app.py",       "pages/06_regional.py" in app_src)
check("page_regional callable removed", "def page_regional" not in app_src)

# ── 2. Data loading ───────────────────────────────────────────────────────
print("\n=== 2. Data loading ===")
t0 = time.time()
master = load_master()
items  = load_item_master()
print(f"  Loaded in {time.time()-t0:.1f}s")

fs     = FilterState()
mf     = apply_master_filters(master, fs)
itf    = apply_item_filters(items, fs)
check("master_f non-empty", len(mf)  > 0, str(len(mf)))
check("items_f non-empty",  len(itf) > 0, str(len(itf)))

# ── 3. sales_by_state ─────────────────────────────────────────────────────
print("\n=== 3. sales_by_state ===")
ss = kpis.sales_by_state(mf)
check("returns DataFrame",              isinstance(ss, pd.DataFrame))
check("has required columns",
      all(c in ss.columns for c in
          ["customer_state","order_count","total_sales","avg_order_value",
           "total_freight","unique_customers"]))
check("27 states",                      len(ss) == 27, str(len(ss)))
check("SP is top state by sales",       ss.iloc[0]["customer_state"] == "SP")
check("total_sales > 0 (all)",          (ss["total_sales"] > 0).all())
check("avg_order_value > 0 (all)",      (ss["avg_order_value"] > 0).all())
check("unique_customers <= order_count (all)",
      (ss["unique_customers"] <= ss["order_count"]).all())

# ── 4. sales_by_seller_state ──────────────────────────────────────────────
print("\n=== 4. sales_by_seller_state ===")
sss = kpis.sales_by_seller_state(itf)
check("returns DataFrame",              isinstance(sss, pd.DataFrame))
check("has required columns",
      all(c in sss.columns for c in
          ["seller_state","item_count","unique_sellers","unique_orders",
           "total_sales_value","avg_item_price","avg_freight_per_item"]))
check("has rows",                       len(sss) > 0, str(len(sss)))
check("SP is top seller state",         sss.iloc[0]["seller_state"] == "SP")
check("total_sales_value > 0 (all)",    (sss["total_sales_value"] > 0).all())
check("unique_sellers >= 1 (all)",      (sss["unique_sellers"] >= 1).all())

# ── 5. delivery_by_state ──────────────────────────────────────────────────
print("\n=== 5. delivery_by_state ===")
ds = kpis.delivery_by_state(mf)
check("returns DataFrame",              isinstance(ds, pd.DataFrame))
check("has required columns",
      all(c in ds.columns for c in
          ["customer_state","order_count","avg_delivery_days",
           "median_delivery_days","delay_rate","n_delayed","n_on_time"]))
check("sorted by avg_delivery_days desc",
      ds["avg_delivery_days"].is_monotonic_decreasing or len(ds) <= 1)
check("delay_rate in [0, 1] (all)",     ds["delay_rate"].between(0, 1).all())
check("avg_delivery_days > 0 (all)",    (ds["avg_delivery_days"] > 0).all())

# ── 6. review_by_state ────────────────────────────────────────────────────
print("\n=== 6. review_by_state ===")
rs = kpis.review_by_state(mf)
check("returns DataFrame",              isinstance(rs, pd.DataFrame))
check("has required columns",
      all(c in rs.columns for c in
          ["customer_state","order_count","avg_review_score",
           "n_reviews","low_score_rate","five_star_rate"]))
check("avg_review_score in [1, 5]",     rs["avg_review_score"].between(1, 5).all())
check("low_score_rate in [0, 1]",       rs["low_score_rate"].between(0, 1).all())
check("five_star_rate in [0, 1]",       rs["five_star_rate"].between(0, 1).all())

# ── 7. KPI scalar helpers ─────────────────────────────────────────────────
print("\n=== 7. KPI scalar helpers ===")
n_states = int(ss["customer_state"].nunique())
check("n_states == 27",                 n_states == 27, str(n_states))

top_seller_state = sss.iloc[0]["seller_state"] if len(sss) > 0 else "N/A"
top_seller_pct = (
    sss.iloc[0]["total_sales_value"] / sss["total_sales_value"].sum() * 100
    if len(sss) > 0 else 0.0
)
check("top_seller_pct in (50, 95)",     50 < top_seller_pct < 95,
      f"{top_seller_pct:.1f}%")

# ── 8. Chart outputs ──────────────────────────────────────────────────────
print("\n=== 8. Chart outputs ===")

fig_orders = charts.bar_horizontal(
    ss.head(15), x="order_count", y="customer_state",
    title="Orders by Customer State",
    x_axis_title="Orders",
)
check("orders bar returns Figure",      is_fig(fig_orders))
check("orders bar has traces",          len(fig_orders.data) > 0)

fig_sales = charts.bar_horizontal(
    ss.head(15), x="total_sales", y="customer_state",
    title="Sales by Customer State",
    x_axis_title="R$",
)
check("sales bar returns Figure",       is_fig(fig_sales))

fig_del = charts.bar_horizontal(
    ds.head(15), x="avg_delivery_days", y="customer_state",
    title="Avg Delivery Days",
    x_axis_title="Days",
)
check("delivery bar returns Figure",    is_fig(fig_del))

fig_rev = charts.bar_horizontal(
    rs.head(15), x="avg_review_score", y="customer_state",
    title="Avg Review Score by State",
    x_axis_title="Score",
)
check("review bar returns Figure",      is_fig(fig_rev))

fig_ss = charts.bar_horizontal(
    sss.head(15), x="item_count", y="seller_state",
    title="Items by Seller State",
    x_axis_title="Items",
)
check("seller state bar returns Figure", is_fig(fig_ss))

# ── 9. kpi_delta_card formatting ─────────────────────────────────────────
print("\n=== 9. kpi_delta_card formatting ===")
for label, val, fmt in [
    ("Active States",          n_states,         "{:,}"),
    ("Top Seller State",       top_seller_state, "{}"),
    ("Top Seller State Share", top_seller_pct,   "{:.1f}%"),
]:
    card = charts.kpi_delta_card(label, val, fmt=fmt)
    check(f"card '{label}' non-empty", len(str(card["value"])) > 0, str(card["value"]))

# ── 10. Grain guards ──────────────────────────────────────────────────────
print("\n=== 10. Data-grain guards ===")
check("master_f has no item price col",      "price"         not in mf.columns)
check("master_f has customer_state",         "customer_state" in mf.columns)
check("items_f has seller_state",            "seller_state"   in itf.columns)
check("items_f has price col",               "price"          in itf.columns)
# Grain independence: master total_sales != items total_sales_value would
# differ because items is filtered by order_month_str (item grain date vs order date)
# but both should be large positive values
ms_total = ss["total_sales"].sum()
it_total = sss["total_sales_value"].sum()
check("master total_sales > 1M",             ms_total > 1_000_000, f"R$ {ms_total:,.0f}")
check("items total_sales_value > 1M",        it_total > 1_000_000, f"R$ {it_total:,.0f}")

# ── 11. Edge case: empty filter ───────────────────────────────────────────
print("\n=== 11. Edge case — empty filter ===")
fs_empty = FilterState(customer_states=["XX_NONE"])
m_empty  = apply_master_filters(master, fs_empty)
i_empty  = apply_item_filters(items, fs_empty)
check("empty master filter: 0 rows",    len(m_empty) == 0)
check("empty items filter: 0 rows",     len(i_empty) == 0)

ss_e  = kpis.sales_by_state(m_empty)
check("sales_by_state(empty) is DF",    isinstance(ss_e, pd.DataFrame))
check("sales_by_state(empty) 0 rows",   len(ss_e) == 0)

ds_e  = kpis.delivery_by_state(m_empty)
check("delivery_by_state(empty) is DF", isinstance(ds_e, pd.DataFrame))

rs_e  = kpis.review_by_state(m_empty)
check("review_by_state(empty) is DF",   isinstance(rs_e, pd.DataFrame))

sss_e = kpis.sales_by_seller_state(i_empty)
check("sales_by_seller_state(empty) DF", isinstance(sss_e, pd.DataFrame))

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED")
    sys.exit(1)
