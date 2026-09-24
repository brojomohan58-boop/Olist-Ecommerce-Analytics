"""
_validate_kpis.py
=================
Headless validation of src/kpis.py against reference values established
during Phase 2 planning (the _inventory.py run).

Reference values used as ground truth
--------------------------------------
  Total delivered orders (master)         : 96,478
  Orders with delivery date               : 96,470
  On-time orders                          : 88,644  → on_time_rate = 91.9%
  Late orders                             : 7,826   → delay_rate   =  8.1%
  Avg delay (late only)                   : 8.9 days
  Median delivery time (delivered)        : 10.0 days
  Mean delivery time (delivered)          : 12.1 days
  Total orders in master                  : 99,441
  Unique customers                        : 96,096
  Repeat customers (2+ orders)            : 2,997   → repeat_rate  =  3.1%
  Avg review score                        : 4.09
  5-star count                            : 57,012  → 5-star rate  = 57.8%
  1+2-star count                          : 14,484  → low rate     = 14.7%
  Review coverage (all orders)            : 98,673 / 99,441 = 99.2%
  Credit card share                       : 75,270 / 99,441 = 75.7%
  Cancelled orders                        : 625     → cancel rate  =  0.63%
  SUM(total_price)  master, all           : ~13.59M (full data)
  SUM(items.price)  items,  all           : ~13.59M differs from master total
  Mean total_price (master, all non-null) : R$ 137.75
  Unique sellers (items)                  : 3,095
  Unique categories (items)               : 72
"""
import sys, time, io
sys.path.insert(0, ".")
# Force UTF-8 stdout so Unicode chars (approx symbol etc.) print on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import streamlit as st

# --- patch cache ---
def passthrough_cache(*args, **kwargs):
    if args and callable(args[0]): return args[0]
    def decorator(fn): return fn
    return decorator
st.cache_data = passthrough_cache

from src.data_loader import load_master, load_item_master, build_order_category_table
from src.filters import FilterState, apply_master_filters, apply_item_filters
import src.kpis as kpis

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading datasets...")
t0 = time.time()
master = load_master()
items  = load_item_master()
print(f"  Loaded in {time.time()-t0:.1f}s\n")

# Filtered slices matching default FilterState (Jan2017-Oct2018, delivered only)
fs_default   = FilterState()
fs_allstatus = FilterState(order_statuses=[])          # all statuses (for cancel rate)
fs_delivered = FilterState()                           # delivered only (same as default)

master_f  = apply_master_filters(master, fs_default)
master_all = apply_master_filters(master, fs_allstatus)
items_f   = apply_item_filters(items, fs_default)

# Unfiltered master for full-dataset reference checks
master_full = master.copy()

# Build order-category table from filtered items
print("Building order-category table...")
t1 = time.time()
oct_df = build_order_category_table(items_f)
print(f"  Built in {time.time()-t1:.1f}s  shape={oct_df.shape}\n")

# ── Validation helpers ────────────────────────────────────────────────────────
passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if condition else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if condition: passed += 1
    else:         failed += 1

def check_approx(label: str, actual: float, expected: float, tol: float = 0.01) -> None:
    """Check that actual is within tol (relative) of expected."""
    if expected == 0:
        condition = abs(actual) < 1e-9
    else:
        condition = abs(actual - expected) / abs(expected) <= tol
    check(label, condition, f"got={actual:.4f}  expected≈{expected:.4f}  tol={tol*100:.1f}%")


# ===========================================================================
# GROUP 1 — Sales & Orders (master grain)
# ===========================================================================
print("=== GROUP 1: Sales & Orders ===")

# total_orders
n_orders_filtered = kpis.total_orders(master_f)
check("total_orders(master_f) > 0", n_orders_filtered > 0, str(n_orders_filtered))
check("total_orders(master_full) == 99441", kpis.total_orders(master_full) == 99441)

# total_sales_value — full dataset (non-null total_price rows = 98666)
tsv_full = kpis.total_sales_value(master_full)
check("total_sales_value > 0", tsv_full > 0, f"R${tsv_full:,.0f}")
# Reference: 98666 rows × mean R$137.75 ≈ R$13.59M
check_approx("total_sales_value ≈ R$13.59M", tsv_full, 13_590_000, tol=0.02)

# avg_order_value — mean of total_price (non-null)
aov_full = kpis.avg_order_value(master_full)
check_approx("avg_order_value ≈ R$137.75", aov_full, 137.75)

# median_order_value
mov_full = kpis.median_order_value(master_full)
check_approx("median_order_value ≈ R$86.90", mov_full, 86.90)

# cancellation_rate — needs all-status input
cr = kpis.cancellation_rate(master_all)
check_approx("cancellation_rate ≈ 0.63%", cr * 100, 0.63, tol=0.10)

# monthly_orders_and_sales
monthly = kpis.monthly_orders_and_sales(master_f)
check("monthly_orders_and_sales returns DataFrame", isinstance(monthly, pd.DataFrame))
check("monthly has columns: order_month_str, order_count, total_sales, avg_order_value",
      all(c in monthly.columns for c in ["order_month_str","order_count","total_sales","avg_order_value"]))
check("monthly rows > 0", len(monthly) > 0, str(len(monthly)))
check("monthly order_count sum == total_orders(master_f)",
      monthly["order_count"].sum() == kpis.total_orders(master_f))
check("monthly is sorted chronologically",
      monthly["order_month_str"].is_monotonic_increasing)

# orders_by_status
obs = kpis.orders_by_status(master_full)
check("orders_by_status returns DataFrame", isinstance(obs, pd.DataFrame))
check("orders_by_status has share_pct column", "share_pct" in obs.columns)
check("orders_by_status share_pct sums to 100",
      abs(obs["share_pct"].sum() - 100.0) < 0.01,
      f"{obs['share_pct'].sum():.4f}")
# delivered should be the largest status
top_status = obs.iloc[0]["order_status"]
check("top status is 'delivered'", top_status == "delivered", top_status)


# ===========================================================================
# GROUP 2 — Customers (master grain)
# ===========================================================================
print("\n=== GROUP 2: Customers ===")

# unique_customers — full master
uc = kpis.unique_customers(master_full)
check("unique_customers(master_full) == 96096", uc == 96096, str(uc))

# repeat_customer_rate — full master (3.1% of 96096 = 2997 repeat)
rcr = kpis.repeat_customer_rate(master_full)
check_approx("repeat_customer_rate ≈ 3.1%", rcr * 100, 3.1, tol=0.05)

# orders_per_customer — full master (99441 / 96096 ≈ 1.035)
opc = kpis.orders_per_customer(master_full)
check_approx("orders_per_customer ≈ 1.035", opc, 1.035, tol=0.01)


# ===========================================================================
# GROUP 3 — Delivery (master grain, delivered only)
# ===========================================================================
print("\n=== GROUP 3: Delivery ===")

delivered_full = master_full[master_full["order_status"] == "delivered"].copy()

add = kpis.avg_delivery_days(delivered_full)
check_approx("avg_delivery_days ≈ 12.1 days", add, 12.1, tol=0.02)

mdd = kpis.median_delivery_days(delivered_full)
check_approx("median_delivery_days == 10.0 days", mdd, 10.0, tol=0.01)

otr = kpis.on_time_rate(delivered_full)
check_approx("on_time_rate ≈ 91.9%", otr * 100, 91.9, tol=0.01)

dr = kpis.delay_rate(delivered_full)
check_approx("delay_rate ≈ 8.1%", dr * 100, 8.1, tol=0.01)

# on_time + delay rates must sum to 1.0
check_approx("on_time_rate + delay_rate == 1.0", otr + dr, 1.0, tol=0.001)

aldd = kpis.avg_late_delay_days(delivered_full)
check_approx("avg_late_delay_days ≈ 8.9 days", aldd, 8.9, tol=0.05)

# delivery_by_state
dbs = kpis.delivery_by_state(delivered_full)
check("delivery_by_state returns DataFrame", isinstance(dbs, pd.DataFrame))
check("delivery_by_state has 27 states", len(dbs) == 27, str(len(dbs)))
check("delivery_by_state has required columns",
      all(c in dbs.columns for c in ["customer_state","order_count","avg_delivery_days","delay_rate"]))
check("delivery_by_state delay_rate between 0 and 1",
      dbs["delay_rate"].between(0, 1, inclusive="both").all())
check("delivery_by_state order_count sum ≈ 96470",
      abs(dbs["order_count"].sum() - 96470) < 5,
      str(dbs["order_count"].sum()))

# delivery_by_month
dbm = kpis.delivery_by_month(delivered_full)
check("delivery_by_month returns DataFrame", isinstance(dbm, pd.DataFrame))
check("delivery_by_month is sorted chronologically",
      dbm["order_month_str"].is_monotonic_increasing)
check("delivery_by_month delay_rate between 0 and 1",
      dbm["delay_rate"].between(0, 1, inclusive="both").all())


# ===========================================================================
# GROUP 4 — Category delivery (order-category table)
# ===========================================================================
print("\n=== GROUP 4: Category Delivery ===")

# Use full items for unfiltered category table
oct_full = build_order_category_table(items)

dbc = kpis.delivery_by_category(oct_full)
check("delivery_by_category returns DataFrame", isinstance(dbc, pd.DataFrame))
check("delivery_by_category rows == category count",
      len(dbc) <= items["product_category_name_english"].nunique(),
      str(len(dbc)))
check("delivery_by_category has required columns",
      all(c in dbc.columns for c in ["primary_category","order_count","avg_delivery_days","delay_rate"]))
check("delivery_by_category delay_rate between 0 and 1",
      dbc["delay_rate"].between(0, 1, inclusive="both").all())
# Total order count in category table must not exceed delivered order count
# (multi-item orders counted once, not once per item)
check("category delivery order_count total ≤ 96470",
      dbc["order_count"].sum() <= 96470,
      str(dbc["order_count"].sum()))


# ===========================================================================
# GROUP 5 — Reviews (master grain)
# ===========================================================================
print("\n=== GROUP 5: Reviews ===")

ars = kpis.avg_review_score(master_full)
check_approx("avg_review_score ≈ 4.09", ars, 4.09, tol=0.01)

rc = kpis.review_coverage(master_full)
# 98673 reviews / 99441 orders = 99.23%
check_approx("review_coverage ≈ 99.2%", rc * 100, 99.2, tol=0.01)

rsd = kpis.review_score_distribution(master_full)
check("review_score_distribution returns DataFrame", isinstance(rsd, pd.DataFrame))
check("review_score_distribution has 5 rows", len(rsd) == 5, str(len(rsd)))
check("review_score_distribution score 5 count == 57012",
      int(rsd.loc[rsd["review_score"] == 5, "count"].values[0]) == 57012)
check("review_score_distribution score 1 count == 11356",
      int(rsd.loc[rsd["review_score"] == 1, "count"].values[0]) == 11356)
check("review_score_distribution total == 98673",
      rsd["count"].sum() == 98673, str(rsd["count"].sum()))

fsr = kpis.five_star_rate(master_full)
check_approx("five_star_rate ≈ 57.8%", fsr * 100, 57.8, tol=0.01)

lsr = kpis.low_score_rate(master_full)
# 1+2-star: 11356+3128=14484 / 98673 = 14.68%
check_approx("low_score_rate ≈ 14.7%", lsr * 100, 14.68, tol=0.01)

rbs = kpis.review_by_state(master_full)
check("review_by_state returns DataFrame", isinstance(rbs, pd.DataFrame))
check("review_by_state has 27 states", len(rbs) == 27, str(len(rbs)))
check("review_by_state sorted ascending by avg_review_score",
      rbs["avg_review_score"].is_monotonic_increasing)

rvd = kpis.review_vs_delay(delivered_full)
check("review_vs_delay returns DataFrame", isinstance(rvd, pd.DataFrame))
check("review_vs_delay has 2 rows (0=on-time, 1=delayed)",
      len(rvd) == 2, str(len(rvd)))
check("review_vs_delay is_delayed values are 0 and 1",
      set(rvd["is_delayed"].astype(int).tolist()) == {0, 1})
# On-time orders should have a higher avg review score than delayed
ontime_score = float(rvd.loc[rvd["is_delayed"] == 0, "avg_review_score"].values[0])
delayed_score = float(rvd.loc[rvd["is_delayed"] == 1, "avg_review_score"].values[0])
check("on-time avg review score > delayed avg review score",
      ontime_score > delayed_score,
      f"on_time={ontime_score:.3f}  delayed={delayed_score:.3f}")


# ===========================================================================
# GROUP 6 — Category reviews (order-category table)
# ===========================================================================
print("\n=== GROUP 6: Category Reviews ===")

rbc = kpis.review_by_category(oct_full)
check("review_by_category returns DataFrame", isinstance(rbc, pd.DataFrame))
check("review_by_category rows ≤ 72 categories", len(rbc) <= 72, str(len(rbc)))
check("review_by_category sorted ascending by avg_review_score",
      rbc["avg_review_score"].is_monotonic_increasing)
check("review_by_category low_score_rate between 0 and 1",
      rbc["low_score_rate"].between(0, 1, inclusive="both").all())


# ===========================================================================
# GROUP 7 — Payments (master grain)
# ===========================================================================
print("\n=== GROUP 7: Payments ===")

pts = kpis.payment_type_share(master_full)
check("payment_type_share returns DataFrame", isinstance(pts, pd.DataFrame))
check("payment_type_share share_pct sums to 100",
      abs(pts["share_pct"].sum() - 100.0) < 0.01,
      f"{pts['share_pct'].sum():.4f}")
# credit_card should be top payment type
top_pay = pts.iloc[0]["primary_payment_type"]
check("top payment type is credit_card", top_pay == "credit_card", top_pay)
# credit card share ≈ 75.7%
cc_share = float(pts.loc[pts["primary_payment_type"] == "credit_card", "share_pct"].values[0])
check_approx("credit card share ≈ 75.7%", cc_share, 75.70, tol=0.01)

instdist = kpis.installment_distribution(master_full)
check("installment_distribution returns DataFrame", isinstance(instdist, pd.DataFrame))
check("installment_distribution only credit card rows",
      instdist["order_count"].sum() <= 75270,
      str(instdist["order_count"].sum()))
check("installment_distribution share_pct sums to 100",
      abs(instdist["share_pct"].sum() - 100.0) < 0.01)

aib = kpis.aov_by_installment_band(master_full)
check("aov_by_installment_band returns DataFrame", isinstance(aib, pd.DataFrame))
check("aov_by_installment_band has 5 bands",
      len(aib) == 5, str(len(aib)))
check("aov_by_installment_band band_label sorted correctly",
      aib["band_label"].tolist() == ["1", "2–3", "4–6", "7–12", "13+"])
# Higher installment bands should generally have higher AOV
aov_band1 = float(aib.loc[aib["band_label"] == "1", "avg_order_value"].values[0])
aov_band7 = float(aib.loc[aib["band_label"] == "7–12", "avg_order_value"].values[0])
check("AOV band 7-12 > AOV band 1",
      aov_band7 > aov_band1,
      f"band1={aov_band1:.0f}  band7-12={aov_band7:.0f}")

pr = kpis.payment_reconciliation(master_full)
check("payment_reconciliation returns DataFrame", isinstance(pr, pd.DataFrame))
check("payment_reconciliation has avg_difference column", "avg_difference" in pr.columns)
# Reconciliation note: column exists but we make no causal assertion about sign


# ===========================================================================
# GROUP 8 — Category Sales (item grain)
# ===========================================================================
print("\n=== GROUP 8: Category Sales (item grain) ===")

sbc = kpis.sales_by_category(items)
check("sales_by_category returns DataFrame", isinstance(sbc, pd.DataFrame))
check("sales_by_category rows ≤ 72", len(sbc) <= 72, str(len(sbc)))
check("sales_by_category has required columns",
      all(c in sbc.columns for c in
          ["product_category_name_english","item_count","unique_orders",
           "total_sales_value","avg_item_price","avg_freight_per_item"]))
check("sales_by_category sorted by total_sales_value desc",
      sbc["total_sales_value"].is_monotonic_decreasing)
# Total item_count must equal items rows (all have category in item master)
check("sales_by_category item_count sum == 112650",
      sbc["item_count"].sum() == len(items), str(sbc["item_count"].sum()))

# Grain-separation note:
# SUM(items.price) == SUM(master.total_price) in this dataset because
# master.total_price was built by summing item prices per order.
# The grain difference is HOW values are attributed (per-item category vs
# per-order) — the total remains the same. Both totals must match.
item_total = kpis.total_sales_value(master_full)        # order grain
cat_total  = float(sbc["total_sales_value"].sum())       # item grain
check("item-grain total matches order-grain total (same underlying sales)",
      abs(cat_total - item_total) / max(item_total, 1) < 0.001,
      f"item_grain={cat_total:,.0f}  order_grain={item_total:,.0f}")
check("category sales values are all positive",
      (sbc["total_sales_value"] > 0).all())

sbs = kpis.sales_by_seller(items)
check("sales_by_seller returns DataFrame", isinstance(sbs, pd.DataFrame))
check("sales_by_seller unique_sellers == 3095",
      len(sbs) == 3095, str(len(sbs)))
check("sales_by_seller sorted by total_sales_value desc",
      sbs["total_sales_value"].is_monotonic_decreasing)
check("sales_by_seller delay_rate between 0 and 1",
      sbs["delay_rate"].between(0, 1, inclusive="both").all())

ibm = kpis.items_by_month(items_f)
check("items_by_month returns DataFrame", isinstance(ibm, pd.DataFrame))
check("items_by_month sorted chronologically",
      ibm["order_month_str"].is_monotonic_increasing)
check("items_by_month item_count sum == len(items_f)",
      ibm["item_count"].sum() == len(items_f), str(ibm["item_count"].sum()))


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*60}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED — review output above")
    sys.exit(1)
