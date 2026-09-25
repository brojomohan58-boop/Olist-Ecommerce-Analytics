"""
Validation script for src/filters.py (headless — no Streamlit session).
Tests FilterState defaults, apply_master_filters, and apply_item_filters.
"""
import sys, time
sys.path.insert(0, ".")

import pandas as pd
import streamlit as st

# Patch cache decorators before any src import
def passthrough_cache(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    def decorator(fn):
        return fn
    return decorator

st.cache_data = passthrough_cache

from src.data_loader import load_master, load_item_master
from src.filters import FilterState, apply_master_filters, apply_item_filters
from datetime import date

print("Loading datasets...")
t0 = time.time()
master = load_master()
items  = load_item_master()
print(f"  Loaded in {time.time()-t0:.1f}s\n")

# ── Helper ────────────────────────────────────────────────────────────────────
passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  PASS  {label}")
        passed += 1
    else:
        print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
        failed += 1


# ── 1. FilterState defaults ───────────────────────────────────────────────────
print("=== 1. FilterState defaults ===")
fs_default = FilterState()
check("date_start == 2017-01-01",  fs_default.date_start == date(2017, 1, 1))
check("date_end   == 2018-10-17",  fs_default.date_end   == date(2018, 10, 17))
check("include_2016 == False",     fs_default.include_2016 is False)
check("order_statuses == ['delivered']", fs_default.order_statuses == ["delivered"])
check("customer_states == []",     fs_default.customer_states == [])
check("seller_states == []",       fs_default.seller_states   == [])
check("categories == []",          fs_default.categories       == [])


# ── 2. apply_master_filters — default FilterState ────────────────────────────
print("\n=== 2. apply_master_filters — default (Jan2017–Oct2018, delivered) ===")
filtered_m = apply_master_filters(master, fs_default)
check("Returns a copy (not same object)", filtered_m is not master)
check("No 2016 partial rows",
      filtered_m["is_2016_partial"].sum() == 0,
      f"found {filtered_m['is_2016_partial'].sum()}")
check("Only 'delivered' status",
      (filtered_m["order_status"] == "delivered").all(),
      str(filtered_m["order_status"].value_counts().to_dict()))
check("All dates >= 2017-01-01",
      (filtered_m["order_purchase_timestamp"] >= pd.Timestamp("2017-01-01")).all())
check("All dates <= 2018-10-17 23:59:59",
      (filtered_m["order_purchase_timestamp"] <= pd.Timestamp("2018-10-17 23:59:59")).all())
print(f"  Rows after default filter: {len(filtered_m):,}  (from {len(master):,} total)")


# ── 3. apply_master_filters — include 2016 ───────────────────────────────────
print("\n=== 3. apply_master_filters — include_2016=True ===")
fs_2016 = FilterState(
    date_start=date(2016, 9, 4),
    date_end=date(2018, 10, 17),
    include_2016=True,
    order_statuses=["delivered"],
)
filtered_2016 = apply_master_filters(master, fs_2016)
n_2016 = filtered_2016["is_2016_partial"].sum()
check("2016 rows present when toggled on",
      n_2016 > 0,
      f"found {n_2016}")
print(f"  2016 rows included: {n_2016}")


# ── 4. apply_master_filters — all statuses ───────────────────────────────────
print("\n=== 4. apply_master_filters — empty order_statuses (= all) ===")
fs_all_status = FilterState(order_statuses=[])
filtered_all = apply_master_filters(master, fs_all_status)
unique_statuses = sorted(filtered_all["order_status"].unique().tolist())
check("Multiple statuses present when order_statuses=[]",
      len(unique_statuses) > 1,
      str(unique_statuses))
print(f"  Statuses present: {unique_statuses}")


# ── 5. apply_master_filters — customer state filter ──────────────────────────
print("\n=== 5. apply_master_filters — customer_states=['SP'] ===")
fs_sp = FilterState(customer_states=["SP"])
filtered_sp = apply_master_filters(master, fs_sp)
check("Only SP rows",
      (filtered_sp["customer_state"] == "SP").all(),
      str(filtered_sp["customer_state"].value_counts().to_dict()))
print(f"  SP rows: {len(filtered_sp):,}")


# ── 6. apply_item_filters — default FilterState ──────────────────────────────
print("\n=== 6. apply_item_filters — default ===")
filtered_i = apply_item_filters(items, fs_default)
check("Returns a copy", filtered_i is not items)
check("Only primary analysis period",
      filtered_i["is_primary_analysis_period"].all(),
      str(filtered_i["is_primary_analysis_period"].value_counts().to_dict()))
check("Only 'delivered' status",
      (filtered_i["order_status"] == "delivered").all())
check("All dates >= 2017-01-01",
      (filtered_i["order_purchase_timestamp"] >= pd.Timestamp("2017-01-01")).all())
print(f"  Rows after default filter: {len(filtered_i):,}  (from {len(items):,} total)")


# ── 7. apply_item_filters — seller state ─────────────────────────────────────
print("\n=== 7. apply_item_filters — seller_states=['SP'] ===")
fs_seller_sp = FilterState(seller_states=["SP"])
filtered_seller = apply_item_filters(items, fs_seller_sp)
check("Only SP seller rows",
      (filtered_seller["seller_state"] == "SP").all())
print(f"  SP seller rows: {len(filtered_seller):,}")


# ── 8. apply_item_filters — category filter ──────────────────────────────────
print("\n=== 8. apply_item_filters — categories=['bed_bath_table'] ===")
fs_cat = FilterState(categories=["bed_bath_table"])
filtered_cat = apply_item_filters(items, fs_cat)
check("Only bed_bath_table rows",
      (filtered_cat["product_category_name_english"] == "bed_bath_table").all())
print(f"  bed_bath_table rows: {len(filtered_cat):,}")


# ── 9. No mutation of originals ───────────────────────────────────────────────
print("\n=== 9. Original DataFrames not mutated ===")
orig_master_len = len(master)
orig_items_len  = len(items)
_ = apply_master_filters(master, FilterState(order_statuses=["canceled"]))
_ = apply_item_filters(items, FilterState(categories=["toys"]))
check("master unchanged", len(master) == orig_master_len)
check("items unchanged",  len(items)  == orig_items_len)


# ── 10. Date boundary edge case ───────────────────────────────────────────────
print("\n=== 10. Single-month window (Jan 2017 only) ===")
fs_jan2017 = FilterState(
    date_start=date(2017, 1, 1),
    date_end=date(2017, 1, 31),
    order_statuses=[],
)
filtered_jan = apply_master_filters(master, fs_jan2017)
check("All dates in Jan 2017",
      (filtered_jan["order_purchase_timestamp"].dt.to_period("M") == "2017-01").all())
print(f"  Jan-2017 orders: {len(filtered_jan):,}")


# ── 11. Category + seller state combined ─────────────────────────────────────
print("\n=== 11. apply_item_filters — combined category + seller state ===")
fs_combo = FilterState(
    categories=["health_beauty"],
    seller_states=["SP"],
    order_statuses=[],
)
filtered_combo = apply_item_filters(items, fs_combo)
check("Only health_beauty",
      (filtered_combo["product_category_name_english"] == "health_beauty").all())
check("Only SP sellers",
      (filtered_combo["seller_state"] == "SP").all())
print(f"  health_beauty + SP seller rows: {len(filtered_combo):,}")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED — review output above")
    sys.exit(1)
