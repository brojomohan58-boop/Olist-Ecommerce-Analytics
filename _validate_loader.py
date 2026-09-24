"""Validation script for src/data_loader.py — run outside Streamlit."""
import sys
sys.path.insert(0, ".")

# Patch st.cache_data to a no-op so the module loads outside Streamlit
import streamlit as st

def passthrough_cache(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    def decorator(fn):
        return fn
    return decorator

st.cache_data = passthrough_cache

from src.data_loader import (
    load_master,
    load_item_master,
    build_order_category_table,
    dataset_summary,
)
import pandas as pd

# ── Dataset summary ────────────────────────────────────────────────────────
print("=== Dataset Summary ===")
for k, v in dataset_summary().items():
    print(f"  {k}: {v}")

# ── Master ─────────────────────────────────────────────────────────────────
print("\n=== Load Master ===")
master = load_master()
print(f"  Shape             : {master.shape}")
print(f"  order_id unique   : {master['order_id'].nunique()}  (must == {len(master)})")
print(f"  year dist         : {master['order_year'].value_counts().sort_index().to_dict()}")
print(f"  is_2016_partial   : {master['is_2016_partial'].value_counts().to_dict()}")
print(f"  is_delayed dtype  : {master['is_delayed'].dtype}")
print(f"  is_delayed dist   : {master['is_delayed'].value_counts(dropna=False).to_dict()}")
print(f"  order_month_str   : {master['order_month_str'].dropna().iloc[:3].tolist()}")
print(f"  NaT in ts         : {master['order_purchase_timestamp'].isna().sum()}")
assert master["order_id"].nunique() == len(master), "FAIL: master grain"
print("  GRAIN OK")

# ── Item Master ────────────────────────────────────────────────────────────
print("\n=== Load Item Master ===")
items = load_item_master()
print(f"  Shape             : {items.shape}")
composite = items.groupby(["order_id", "order_item_id"]).ngroups
print(f"  Composite unique  : {composite}  (must == {len(items)})")
print(f"  is_delayed dtype  : {items['is_delayed'].dtype}")
print(f"  is_primary_period : {items['is_primary_analysis_period'].value_counts().to_dict()}")
top5 = items["product_category_name_english"].value_counts().head(5).to_dict()
print(f"  Top-5 categories  : {top5}")
assert composite == len(items), "FAIL: item master grain"
print("  GRAIN OK")

# ── Order-Category Table ───────────────────────────────────────────────────
print("\n=== Build Order Category Table ===")
oct_df = build_order_category_table(items)
print(f"  Shape             : {oct_df.shape}")
print(f"  order_id unique   : {oct_df['order_id'].nunique()}  (must == {len(oct_df)})")
print(f"  Columns           : {list(oct_df.columns)}")
multi_cat = oct_df["has_multiple_categories"].value_counts().to_dict()
print(f"  multi-cat dist    : {multi_cat}")
n_cat_dist = oct_df["n_categories"].value_counts().sort_index().to_dict()
print(f"  n_cat distribution: {n_cat_dist}")
assert oct_df["order_id"].nunique() == len(oct_df), "FAIL: category table grain"
print("  GRAIN OK")

# Verify tie-break determinism: run twice and compare
oct_df2 = build_order_category_table(items)
assert oct_df["primary_category"].equals(oct_df2["primary_category"]), \
    "FAIL: category table is not deterministic"
print("  DETERMINISM OK")

# Show sample multi-category orders
multi = oct_df[oct_df["has_multiple_categories"]]
print(f"\n  Multi-category orders ({len(multi)} total) — first 5:")
print(multi[["order_id", "primary_category", "n_categories"]].head(5).to_string(index=False))

print("\n\nALL VALIDATIONS PASSED")
