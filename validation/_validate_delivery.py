"""_validate_delivery.py — headless validation of pages/03_delivery_operations.py."""
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
for f in ["pages/03_delivery_operations.py", "app.py"]:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        check(f"Syntax OK: {f}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {f}", False, str(e))

app_src = open("app.py", encoding="utf-8").read()
check("03_delivery_operations.py in app.py",  "pages/03_delivery_operations.py" in app_src)
check("page_delivery callable removed",        "def page_delivery" not in app_src)

# ── 2. Data loading & filtering ───────────────────────────────────────────
print("\n=== 2. Data loading ===")
t0 = time.time()
master = load_master()
items  = load_item_master()
print(f"  Loaded in {time.time()-t0:.1f}s")

fs        = FilterState()                                    # delivered, Jan2017-Oct2018
fs_all    = FilterState(order_statuses=[])
master_f  = apply_master_filters(master, fs)
master_all = apply_master_filters(master, fs_all)
items_f   = apply_item_filters(items, fs)
check("master_f non-empty",   len(master_f) > 0,  str(len(master_f)))
check("items_f non-empty",    len(items_f)  > 0,  str(len(items_f)))

# ── 3. Overall KPI scalars ────────────────────────────────────────────────
print("\n=== 3. Overall KPI scalars ===")
avg_del  = kpis.avg_delivery_days(master_f)
med_del  = kpis.median_delivery_days(master_f)
otr      = kpis.on_time_rate(master_f)
dr       = kpis.delay_rate(master_f)
avg_late = kpis.avg_late_delay_days(master_f)
cancel_r = kpis.cancellation_rate(master_all)

check("avg_delivery in [5,30]",      5  < avg_del  < 30,  f"{avg_del:.2f}")
check("median_delivery in [5,20]",   5  < med_del  < 20,  f"{med_del:.2f}")
check("on_time_rate in (0,1)",        0  < otr      < 1,   f"{otr:.4f}")
check("delay_rate in (0,1)",          0  < dr       < 1,   f"{dr:.4f}")
check("otr + dr == 1.0",             abs(otr + dr - 1.0) < 1e-6)
check("avg_late > 0",                avg_late > 0,          f"{avg_late:.2f}")
check("cancel_rate < 5%",            cancel_r < 0.05,       f"{cancel_r:.4f}")

n_del     = int(master_f["delivery_time_days"].notna().sum())
n_delayed = int((master_f["is_delayed"] == 1).sum())
check("n_del > 0",        n_del     > 0, str(n_del))
check("n_delayed > 0",    n_delayed > 0, str(n_delayed))
check("n_delayed < n_del", n_delayed < n_del)

# ── 4. delivery_by_state ─────────────────────────────────────────────────
print("\n=== 4. delivery_by_state ===")
dbs = kpis.delivery_by_state(master_f)
check("returns DataFrame",             isinstance(dbs, pd.DataFrame))
check("27 states",                     len(dbs) == 27, str(len(dbs)))
check("required columns present",
      all(c in dbs.columns for c in
          ["customer_state","order_count","avg_delivery_days","delay_rate"]))
check("delay_rate in [0,1]",           dbs["delay_rate"].between(0,1).all())
check("avg_delivery_days > 0",         (dbs["avg_delivery_days"] > 0).all())
check("order_count sum ≈ n_del",
      abs(dbs["order_count"].sum() - n_del) <= 5,
      str(dbs["order_count"].sum()))

# delay_rate % column for chart
dbs_pct = dbs.copy()
dbs_pct["delay_rate_pct"] = dbs_pct["delay_rate"] * 100
check("delay_rate_pct in [0,100]",     dbs_pct["delay_rate_pct"].between(0,100).all())

# ── 5. delivery_by_month ─────────────────────────────────────────────────
print("\n=== 5. delivery_by_month ===")
dbm = kpis.delivery_by_month(master_f)
check("returns DataFrame",             isinstance(dbm, pd.DataFrame))
check("sorted chronologically",        dbm["order_month_str"].is_monotonic_increasing)
check("delay_rate in [0,1]",           dbm["delay_rate"].between(0,1).all())
check("avg_delivery_days > 0",         (dbm["avg_delivery_days"] > 0).all())

# ── 6. build_order_category_table + delivery_by_category ─────────────────
print("\n=== 6. Category delivery (order-category table) ===")
t1 = time.time()
oct_df = build_order_category_table(items_f)
print(f"  oct built in {time.time()-t1:.1f}s  shape={oct_df.shape}")

# Grain: one row per order
check("oct_df grain: unique order_ids",
      oct_df["order_id"].nunique() == len(oct_df), str(len(oct_df)))

# multi-category flag
n_multi = int(oct_df["has_multiple_categories"].sum())
check("has_multiple_categories flag present", "has_multiple_categories" in oct_df.columns)
check("n_multi >= 0",                          n_multi >= 0, str(n_multi))

dbc = kpis.delivery_by_category(oct_df)
check("dbc returns DataFrame",                 isinstance(dbc, pd.DataFrame))
check("dbc rows <= 72",                        len(dbc) <= 72, str(len(dbc)))
check("dbc delay_rate in [0,1]",               dbc["delay_rate"].between(0,1).all())
check("dbc avg_delivery_days > 0",             (dbc["avg_delivery_days"] > 0).all())
# No double-counting: sum of category orders must not exceed total delivered orders
check("category order_count total <= n_del",
      dbc["order_count"].sum() <= n_del,
      str(dbc["order_count"].sum()))

# delay_rate_pct column
dbc_pct = dbc.copy()
dbc_pct["delay_rate_pct"] = dbc_pct["delay_rate"] * 100
check("dbc delay_rate_pct in [0,100]",         dbc_pct["delay_rate_pct"].between(0,100).all())

# ── 7. Chart outputs ──────────────────────────────────────────────────────
print("\n=== 7. Chart outputs ===")

# Histogram
fig_hist = charts.histogram(
    master_f["delivery_time_days"].dropna(),
    title="Delivery Time",
    xaxis_title="Days", nbins=40, clip_upper=60,
)
check("histogram returns Figure",              is_fig(fig_hist))
check("histogram has 1 trace",                 len(fig_hist.data) == 1)
check("clip annotation present (max>=60)",
      any("clipped" in str(a.text).lower() for a in fig_hist.layout.annotations))

# On-time/late donut
n_ontime  = int((master_f["is_delayed"] == 0).sum())
fig_donut = charts.donut_chart(
    labels=["On-Time", "Late"], values=[n_ontime, n_delayed],
    title="On-Time vs Late",
)
check("donut returns Figure",                  is_fig(fig_donut))
check("donut values sum == n_del",
      sum(fig_donut.data[0].values) == n_ontime + n_delayed)

# Monthly line trend
fig_monthly = charts.line_trend(
    dbm, x="order_month_str",
    y_cols=["avg_delivery_days", "delay_rate"],
    title="Monthly Delivery",
    secondary_y_col="delay_rate",
    y_axis_title="Days", y2_axis_title="Delay Rate",
    add_delay_note=True,
)
check("monthly trend returns Figure",          is_fig(fig_monthly))
check("monthly trend has 2 traces",            len(fig_monthly.data) == 2)
check("delay note on monthly chart",
      any(charts._DELAY_NOTE[:20] in str(a.text) for a in fig_monthly.layout.annotations))

# State bars
fig_state_del = charts.bar_horizontal(dbs, x="avg_delivery_days",
                                       y="customer_state",
                                       title="State Delivery", top_n=27)
check("state delivery bar returns Figure",     is_fig(fig_state_del))
check("27 state bars",                         len(fig_state_del.data[0].y) == 27)

fig_state_dr = charts.bar_horizontal(dbs_pct, x="delay_rate_pct",
                                      y="customer_state",
                                      title="State Delay %", top_n=27)
check("state delay rate bar returns Figure",   is_fig(fig_state_dr))

# Category bars
for top_n in [5, 20]:
    fig_cat = charts.bar_horizontal(dbc, x="avg_delivery_days",
                                     y="primary_category",
                                     title=f"Cat del top{top_n}", top_n=top_n,
                                     add_multicategory_note=True)
    actual = len(fig_cat.data[0].y)
    expected = min(top_n, len(dbc))
    check(f"category bar top_n={top_n}: bars={actual}",
          actual == expected, f"expected={expected}")

# ── 8. kpi_delta_card checks ─────────────────────────────────────────────
print("\n=== 8. kpi_delta_card formatting ===")
for label, val, fmt in [
    ("Avg Delivery Time",      avg_del,      "{:.1f} days"),
    ("On-Time Delivery Rate",  otr * 100,    "{:.1f}%"),
    ("Delay Rate",             dr * 100,     "{:.1f}%"),
    ("Avg Delay (late orders)",avg_late,     "{:.1f} days"),
]:
    card = charts.kpi_delta_card(label, val, fmt=fmt)
    check(f"card '{label}' non-empty", len(card["value"]) > 0, card["value"])

# ── 9. Edge case: empty filter ────────────────────────────────────────────
print("\n=== 9. Edge case — empty filter ===")
fs_empty    = FilterState(customer_states=["XX_NONE"])
master_empty = apply_master_filters(master, fs_empty)
items_empty  = apply_item_filters(items, fs_empty)
check("empty master",                len(master_empty) == 0)
check("avg_delivery_days(empty) == 0.0", kpis.avg_delivery_days(master_empty) == 0.0)
check("on_time_rate(empty) == 0.0",      kpis.on_time_rate(master_empty) == 0.0)
dbs_empty = kpis.delivery_by_state(master_empty)
check("delivery_by_state(empty) DF",     isinstance(dbs_empty, pd.DataFrame))
dbm_empty = kpis.delivery_by_month(master_empty)
check("delivery_by_month(empty) DF",     isinstance(dbm_empty, pd.DataFrame))
oct_empty = build_order_category_table(items_empty)
check("oct_df(empty): 0 rows",           len(oct_empty) == 0)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED")
    sys.exit(1)
