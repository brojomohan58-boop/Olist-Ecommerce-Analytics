"""_validate_customer.py — headless validation of pages/04_customer_experience.py."""
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
for f in ["pages/04_customer_experience.py", "app.py"]:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        check(f"Syntax OK: {f}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {f}", False, str(e))

app_src = open("app.py", encoding="utf-8").read()
check("04_customer_experience.py in app.py",  "pages/04_customer_experience.py" in app_src)
check("page_customer callable removed",        "def page_customer" not in app_src)

# ── 2. Data loading ───────────────────────────────────────────────────────
print("\n=== 2. Data loading ===")
t0 = time.time()
master = load_master()
items  = load_item_master()
print(f"  Loaded in {time.time()-t0:.1f}s")
fs       = FilterState()
master_f = apply_master_filters(master, fs)
items_f  = apply_item_filters(items, fs)
check("master_f non-empty", len(master_f) > 0, str(len(master_f)))

# ── 3. Review KPI scalars ─────────────────────────────────────────────────
print("\n=== 3. Review KPI scalars ===")
avg_rev  = kpis.avg_review_score(master_f)
five_s   = kpis.five_star_rate(master_f)
low_s    = kpis.low_score_rate(master_f)
rev_cov  = kpis.review_coverage(master_f)
n_reviews = int(master_f["review_score"].notna().sum())
n_orders  = kpis.total_orders(master_f)

check("avg_review in [1,5]",     1 <= avg_rev <= 5,     f"{avg_rev:.3f}")
check("five_star_rate in (0,1)", 0 < five_s < 1,        f"{five_s:.4f}")
check("low_score_rate in (0,1)", 0 < low_s < 1,         f"{low_s:.4f}")
check("rev_cov > 0.95",         rev_cov > 0.95,         f"{rev_cov:.4f}")
check("n_reviews <= n_orders",  n_reviews <= n_orders)
check("n_reviews > 0",          n_reviews > 0,           str(n_reviews))

# known reference values
check("avg_review approx 4.09-4.20",  4.0 < avg_rev < 4.3, f"{avg_rev:.3f}")
check("five_star > 0.55",             five_s > 0.55)
check("low_score < 0.20",             low_s  < 0.20)

# ── 4. review_score_distribution ─────────────────────────────────────────
print("\n=== 4. review_score_distribution ===")
rsd = kpis.review_score_distribution(master_f)
check("returns DataFrame",            isinstance(rsd, pd.DataFrame))
check("has 5 rows",                   len(rsd) == 5, str(len(rsd)))
check("scores are 1-5",               rsd["review_score"].tolist() == [1,2,3,4,5])
check("count sum == n_reviews",       rsd["count"].sum() == n_reviews, str(rsd["count"].sum()))
check("share_pct sums to 100",        abs(rsd["share_pct"].sum() - 100.0) < 0.01)
check("all counts >= 0",              (rsd["count"] >= 0).all())

# ── 5. monthly review trend ───────────────────────────────────────────────
print("\n=== 5. Monthly review trend ===")
monthly_rev = (
    master_f.dropna(subset=["review_score"])
    .groupby("order_month_str", sort=True)
    .agg(avg_review_score=("review_score", "mean"),
         n_reviews=("review_score", "count"))
    .reset_index()
)
check("monthly_rev is DataFrame",     isinstance(monthly_rev, pd.DataFrame))
check("monthly_rev sorted",           monthly_rev["order_month_str"].is_monotonic_increasing)
check("avg_review_score in [1,5]",    monthly_rev["avg_review_score"].between(1, 5).all())
check("n_reviews sum == n_reviews",   monthly_rev["n_reviews"].sum() == n_reviews)

# ── 6. review_vs_delay ────────────────────────────────────────────────────
print("\n=== 6. review_vs_delay ===")
rvd = kpis.review_vs_delay(master_f)
check("returns DataFrame",            isinstance(rvd, pd.DataFrame))
check("has 2 rows",                   len(rvd) == 2, str(len(rvd)))
check("is_delayed values are 0,1",    set(rvd["is_delayed"].astype(int).tolist()) == {0,1})
check("on-time avg score > delayed",
      float(rvd.loc[rvd["is_delayed"]==0,"avg_review_score"].values[0]) >
      float(rvd.loc[rvd["is_delayed"]==1,"avg_review_score"].values[0]))
check("low_score_rate in [0,1]",      rvd["low_score_rate"].between(0,1).all())
check("five_star_rate in [0,1]",      rvd["five_star_rate"].between(0,1).all())

# delivery_status label mapping (done in the page)
rvd_display = rvd.copy()
rvd_display["delivery_status"] = rvd_display["is_delayed"].map(
    {0: "On-Time", 1: "Delayed"}
).fillna(rvd_display["is_delayed"].astype(str))
check("delivery_status labels correct",
      set(rvd_display["delivery_status"].tolist()) == {"On-Time", "Delayed"})

# ── 7. review_by_state ────────────────────────────────────────────────────
print("\n=== 7. review_by_state ===")
rbs = kpis.review_by_state(master_f)
check("returns DataFrame",            isinstance(rbs, pd.DataFrame))
check("27 states",                    len(rbs) == 27, str(len(rbs)))
check("sorted ascending by avg_review_score",
      rbs["avg_review_score"].is_monotonic_increasing)
check("low_score_rate in [0,1]",      rbs["low_score_rate"].between(0,1).all())
check("five_star_rate in [0,1]",      rbs["five_star_rate"].between(0,1).all())

rbs_pct = rbs.copy()
rbs_pct["low_score_pct"] = rbs_pct["low_score_rate"] * 100
check("low_score_pct in [0,100]",     rbs_pct["low_score_pct"].between(0,100).all())

# ── 8. review_by_category (order-category table) ─────────────────────────
print("\n=== 8. review_by_category (oct) ===")
t1 = time.time()
oct_df = build_order_category_table(items_f)
print(f"  oct built in {time.time()-t1:.1f}s  shape={oct_df.shape}")
check("oct grain: unique order_ids",  oct_df["order_id"].nunique() == len(oct_df))

rbc = kpis.review_by_category(oct_df)
check("returns DataFrame",            isinstance(rbc, pd.DataFrame))
check("rows <= 72",                   len(rbc) <= 72, str(len(rbc)))
check("sorted ascending by avg_review_score",
      rbc["avg_review_score"].is_monotonic_increasing)
check("low_score_rate in [0,1]",      rbc["low_score_rate"].between(0,1).all())
check("five_star_rate in [0,1]",      rbc["five_star_rate"].between(0,1).all())
check("n_reviews > 0 per category",   (rbc["n_reviews"] > 0).all())

# low_score_pct column (used in page)
rbc_pct = rbc.copy()
rbc_pct["low_score_pct"] = rbc_pct["low_score_rate"] * 100
check("rbc low_score_pct in [0,100]", rbc_pct["low_score_pct"].between(0,100).all())

# ── 9. Chart outputs ──────────────────────────────────────────────────────
print("\n=== 9. Chart outputs ===")
# Review distribution bar
fig_dist = charts.bar_vertical(
    rsd, x="review_score", y="count",
    title="Review Score Distribution",
    x_axis_title="Score", y_axis_title="Orders",
)
check("dist bar returns Figure",      is_fig(fig_dist))
check("dist bar has traces",          len(fig_dist.data) > 0)

# Monthly trend line
fig_trend = charts.line_trend(
    monthly_rev, x="order_month_str",
    y_cols=["avg_review_score"],
    title="Monthly Avg Review Score",
    y_axis_title="Avg Score",
)
check("monthly trend returns Figure", is_fig(fig_trend))
check("monthly trend has 1 trace",    len(fig_trend.data) == 1)

# Review vs delay grouped bar
fig_rvd = charts.grouped_bar(
    rvd_display, x="delivery_status",
    y_cols=["avg_review_score"],
    title="Avg Review Score by Delivery Status",
    y_labels={"avg_review_score": "Avg Review Score"},
)
check("review vs delay bar Figure",   is_fig(fig_rvd))
check("review vs delay 2 x-values",  len(fig_rvd.data[0].x) == 2)

# State bar (avg review, sorted ascending = worst first)
fig_state = charts.bar_horizontal(
    rbs, x="avg_review_score", y="customer_state",
    title="State Review", top_n=27,
)
check("state review bar Figure",      is_fig(fig_state))
check("27 state bars",                len(fig_state.data[0].y) == 27)

# Category bars
for top_n in [5, 20]:
    fig_cat = charts.bar_horizontal(
        rbc, x="avg_review_score", y="primary_category",
        title=f"Cat review top{top_n}", top_n=top_n,
        add_multicategory_note=True,
    )
    actual = len(fig_cat.data[0].y)
    expected = min(top_n, len(rbc))
    check(f"category review bar top_n={top_n}: bars={actual}",
          actual == expected, f"expected={expected}")

# ── 10. No master-grain columns on items & vice versa ────────────────────
print("\n=== 10. Data-grain guards ===")
check("master_f has no price column",         "price"          not in master_f.columns)
check("master_f has no freight_value column", "freight_value"  not in master_f.columns)
check("master_f has review_score column",     "review_score"   in master_f.columns)
check("master_f has is_delayed column",       "is_delayed"     in master_f.columns)

# ── 11. Edge cases ────────────────────────────────────────────────────────
print("\n=== 11. Edge cases — empty filter ===")
fs_empty    = FilterState(customer_states=["XX_NONE"])
master_empty = apply_master_filters(master, fs_empty)
items_empty  = apply_item_filters(items, fs_empty)

check("empty master: 0 rows",                 len(master_empty) == 0)
check("avg_review_score(empty) == 0.0",       kpis.avg_review_score(master_empty) == 0.0)
check("review_coverage(empty) == 0.0",        kpis.review_coverage(master_empty) == 0.0)
check("five_star_rate(empty) == 0.0",         kpis.five_star_rate(master_empty) == 0.0)
check("low_score_rate(empty) == 0.0",         kpis.low_score_rate(master_empty) == 0.0)

rsd_empty = kpis.review_score_distribution(master_empty)
check("rsd(empty): 5 rows all 0",
      len(rsd_empty) == 5 and rsd_empty["count"].sum() == 0)

rbs_empty = kpis.review_by_state(master_empty)
check("review_by_state(empty) is DF",         isinstance(rbs_empty, pd.DataFrame))

rvd_empty = kpis.review_vs_delay(master_empty)
check("review_vs_delay(empty) is DF",         isinstance(rvd_empty, pd.DataFrame))

oct_empty = build_order_category_table(items_empty)
check("oct(empty): 0 rows",                   len(oct_empty) == 0)

rbc_empty = kpis.review_by_category(oct_empty)
check("review_by_category(empty) is DF",      isinstance(rbc_empty, pd.DataFrame))

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED")
    sys.exit(1)
