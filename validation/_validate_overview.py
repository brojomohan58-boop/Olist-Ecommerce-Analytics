"""_validate_overview.py — headless validation of pages/01_overview.py."""
import sys, io, ast
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
from src.filters import FilterState, apply_master_filters
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


# ── 1. Syntax check ───────────────────────────────────────────────────────
print("=== 1. Syntax ===")
for f in ["pages/01_overview.py", "app.py"]:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        check(f"Syntax OK: {f}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {f}", False, str(e))


# ── 2. app.py wiring ──────────────────────────────────────────────────────
print("\n=== 2. app.py wiring ===")
app_src = open("app.py", encoding="utf-8").read()
check("01_overview.py referenced in app.py",
      "pages/01_overview.py" in app_src)
check("page_overview callable removed from app.py",
      "def page_overview" not in app_src)
check("st.Page file-ref used",
      'st.Page("pages/01_overview.py"' in app_src)


# ── 3. Load data ──────────────────────────────────────────────────────────
print("\n=== 3. Data loading ===")
import time
t0 = time.time()
master = load_master()
items  = load_item_master()
print(f"  Loaded in {time.time()-t0:.1f}s")
check("master loaded", len(master) == 99441, str(len(master)))
check("items loaded",  len(items)  == 112650, str(len(items)))

fs        = FilterState()                          # default: delivered, Jan2017-Oct2018
fs_all    = FilterState(order_statuses=[])         # all statuses
master_f  = apply_master_filters(master, fs)
master_all = apply_master_filters(master, fs_all)
check("master_f has rows",  len(master_f)  > 0, str(len(master_f)))
check("master_all has rows", len(master_all) > 0, str(len(master_all)))


# ── 4. KPI computations ───────────────────────────────────────────────────
print("\n=== 4. KPI computations ===")

n_orders    = kpis.total_orders(master_f)
sales_val   = kpis.total_sales_value(master_f)
freight_val = kpis.total_freight_value(master_f)
aov         = kpis.avg_order_value(master_f)
median_ov   = kpis.median_order_value(master_f)
n_customers = kpis.unique_customers(master_f)
cancel_rate = kpis.cancellation_rate(master_all)

check("total_orders > 0",    n_orders > 0,    str(n_orders))
check("total_sales > 0",     sales_val > 0,   f"R${sales_val:,.0f}")
check("total_freight > 0",   freight_val > 0, f"R${freight_val:,.0f}")
check("AOV in range",        80 < aov < 200,  f"R${aov:.2f}")
check("median_ov in range",  50 < median_ov < 150, f"R${median_ov:.2f}")
check("unique_customers > 0", n_customers > 0, str(n_customers))
check("cancel_rate in [0,1]", 0 <= cancel_rate <= 1, f"{cancel_rate:.4f}")
check("cancel_rate < 5%",     cancel_rate < 0.05)

otr      = kpis.on_time_rate(master_f)
dr       = kpis.delay_rate(master_f)
avg_del  = kpis.avg_delivery_days(master_f)
med_del  = kpis.median_delivery_days(master_f)
avg_late = kpis.avg_late_delay_days(master_f)
avg_rev  = kpis.avg_review_score(master_f)
five_s   = kpis.five_star_rate(master_f)
low_s    = kpis.low_score_rate(master_f)
rev_cov  = kpis.review_coverage(master_f)

check("on_time_rate in (0,1)",     0 < otr < 1,    f"{otr:.4f}")
check("delay_rate in (0,1)",       0 < dr < 1,     f"{dr:.4f}")
check("otr + dr == 1.0",           abs(otr + dr - 1.0) < 1e-6)
check("avg_delivery in range",     5 < avg_del < 30, f"{avg_del:.2f}")
check("median_delivery in range",  5 < med_del < 20, f"{med_del:.2f}")
check("avg_late > 0",              avg_late > 0,   f"{avg_late:.2f}")
check("avg_review in [1,5]",       1 <= avg_rev <= 5, f"{avg_rev:.3f}")
check("five_star_rate in (0,1)",   0 < five_s < 1)
check("low_score_rate in (0,1)",   0 < low_s < 1)
check("review_coverage > 0.95",    rev_cov > 0.95, f"{rev_cov:.4f}")

# Denominator transparency checks
n_with_delivery = int(master_f["delivery_time_days"].notna().sum())
n_with_review   = int(master_f["review_score"].notna().sum())
n_late          = int((master_f["is_delayed"] == 1).sum())
check("n_with_delivery > 0",    n_with_delivery > 0, str(n_with_delivery))
check("n_with_review > 0",      n_with_review > 0,   str(n_with_review))
check("n_late > 0",             n_late > 0,           str(n_late))
check("n_with_delivery <= n_orders", n_with_delivery <= n_orders)


# ── 5. Monthly trend DataFrame ────────────────────────────────────────────
print("\n=== 5. Monthly trend ===")
monthly = kpis.monthly_orders_and_sales(master_f)
check("monthly returns DataFrame",     isinstance(monthly, pd.DataFrame))
check("monthly has > 0 rows",          len(monthly) > 0, str(len(monthly)))
check("monthly sorted chronologically", monthly["order_month_str"].is_monotonic_increasing)
check("order_count sum == total_orders",
      monthly["order_count"].sum() == n_orders)
check("total_sales sum matches kpis.total_sales_value",
      abs(monthly["total_sales"].sum() - sales_val) < 1.0,
      f"monthly={monthly['total_sales'].sum():.2f}  kpi={sales_val:.2f}")
check("avg_order_value column present", "avg_order_value" in monthly.columns)


# ── 6. Chart outputs ──────────────────────────────────────────────────────
print("\n=== 6. Chart outputs ===")

fig_trend = charts.line_trend(
    monthly,
    x="order_month_str",
    y_cols=["order_count", "total_sales"],
    title="Monthly Orders & Sales Value",
    secondary_y_col="total_sales",
    y_axis_title="Orders",
    y2_axis_title="Sales Value (R$)",
)
check("line_trend returns Figure",  is_fig(fig_trend))
check("line_trend has 2 traces",    len(fig_trend.data) == 2)
check("secondary y axis set",       fig_trend.data[1].yaxis == "y2")

fig_aov = charts.line_trend(
    monthly,
    x="order_month_str",
    y_cols=["avg_order_value"],
    title="Monthly Average Order Value (R$)",
    y_axis_title="R$",
)
check("AOV trend returns Figure",   is_fig(fig_aov))
check("AOV trend has 1 trace",      len(fig_aov.data) == 1)

obs = kpis.orders_by_status(master_all)
check("orders_by_status has rows",  len(obs) > 0)
check("share_pct sums to 100",      abs(obs["share_pct"].sum() - 100.0) < 0.01)

fig_status = charts.donut_chart(
    labels=obs["order_status"].tolist(),
    values=obs["order_count"].tolist(),
    title="Order Status Share",
)
check("status donut returns Figure", is_fig(fig_status))
check("status donut has Pie trace",  isinstance(fig_status.data[0], go.Pie))
check("status donut labels count",   len(fig_status.data[0].labels) == len(obs))


# ── 7. kpi_delta_card formatting ─────────────────────────────────────────
print("\n=== 7. kpi_delta_card formatting ===")
card = charts.kpi_delta_card("Total Orders", n_orders, fmt="{:,.0f}")
check("card is dict",         isinstance(card, dict))
check("card value non-empty", len(card["value"]) > 0)
check("card label == 'Total Orders'", card["label"] == "Total Orders")

pct_card = charts.kpi_delta_card("On-Time", otr * 100, fmt="{:.1f}%")
check("percent format works", pct_card["value"].endswith("%"))

r_card = charts.kpi_delta_card("AOV", aov, fmt="R$ {:,.2f}")
check("R$ format works",      r_card["value"].startswith("R$"))


# ── 8. No double-counting guard ───────────────────────────────────────────
print("\n=== 8. Data-grain guards ===")
# Page uses master only — confirm no item-level columns were accidentally pulled
item_only_cols = {"price", "freight_value", "order_item_id"}
check("master_f has no item-grain columns",
      not any(c in master_f.columns for c in item_only_cols))

# Order count in monthly trend must not exceed master_f row count
check("monthly order_count never inflated",
      monthly["order_count"].sum() == len(master_f))

# Delivery denominator must be <= total orders
check("delivery denominator <= total orders",
      n_with_delivery <= n_orders)

# Review denominator must be <= total orders
check("review denominator <= total orders",
      n_with_review <= n_orders)


# ── 9. Edge case: empty filtered DataFrame ────────────────────────────────
print("\n=== 9. Edge case — empty filter ===")
# Force an empty result with an impossible filter
fs_empty   = FilterState(customer_states=["XX_NONEXISTENT"])
master_empty = apply_master_filters(master, fs_empty)
check("empty filter produces 0 rows", len(master_empty) == 0)
# KPIs on empty DataFrame should return 0 / 0.0, not raise
check("total_orders(empty) == 0",       kpis.total_orders(master_empty) == 0)
check("total_sales_value(empty) == 0.0", kpis.total_sales_value(master_empty) == 0.0)
check("on_time_rate(empty) == 0.0",      kpis.on_time_rate(master_empty) == 0.0)
check("avg_review_score(empty) == 0.0",  kpis.avg_review_score(master_empty) == 0.0)
monthly_empty = kpis.monthly_orders_and_sales(master_empty)
check("monthly_orders_and_sales(empty) returns empty DF",
      isinstance(monthly_empty, pd.DataFrame) and len(monthly_empty) == 0)


# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED")
    sys.exit(1)
