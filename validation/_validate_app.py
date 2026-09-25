"""
Validation script for app.py (Phase 1 completion check).
Validates: syntax, import chain, data contract integrity, no accidental merge.
Does NOT start a Streamlit server.
"""
import sys, ast, time, importlib.util
from pathlib import Path
sys.path.insert(0, ".")

passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    if condition:
        passed += 1
    else:
        failed += 1


# ── 1. Syntax check on all Phase 1 files ─────────────────────────────────────
print("=== 1. Syntax validation ===")
files_to_check = [
    "app.py",
    "src/data_loader.py",
    "src/filters.py",
    "src/__init__.py",
]
for fpath in files_to_check:
    src_text = Path(fpath).read_text(encoding="utf-8")
    try:
        ast.parse(src_text)
        check(f"Syntax OK: {fpath}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {fpath}", False, str(e))


# ── 2. Import chain: data_loader and filters load cleanly ────────────────────
print("\n=== 2. Import chain ===")
import streamlit as st

def passthrough_cache(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    def decorator(fn):
        return fn
    return decorator
st.cache_data = passthrough_cache

try:
    from src.data_loader import (
        load_master, load_item_master,
        build_order_category_table, dataset_summary,
        ANALYSIS_START, ANALYSIS_END,
    )
    check("src.data_loader imports cleanly", True)
except Exception as e:
    check("src.data_loader imports cleanly", False, str(e))

try:
    from src.filters import (
        FilterState, render_sidebar_filters,
        apply_master_filters, apply_item_filters,
    )
    check("src.filters imports cleanly", True)
except Exception as e:
    check("src.filters imports cleanly", False, str(e))


# ── 3. app.py imports its dependencies without error ─────────────────────────
print("\n=== 3. app.py dependency imports ===")
# Parse app.py and extract all 'from X import Y' and 'import X' statements
app_src = Path("app.py").read_text(encoding="utf-8")
tree = ast.parse(app_src)
imports_found = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        imports_found.append(f"from {node.module} import ...")
    elif isinstance(node, ast.Import):
        for alias in node.names:
            imports_found.append(f"import {alias.name}")
print(f"  Import statements in app.py: {imports_found}")

check("app.py imports streamlit",        "import streamlit" in imports_found)
check("app.py imports src.data_loader",  any("src.data_loader" in i for i in imports_found))
check("app.py imports src.filters",      any("src.filters" in i for i in imports_found))


# ── 4. Data loading and grain assertions ─────────────────────────────────────
print("\n=== 4. Data loading and grain assertions ===")
t0 = time.time()
master = load_master()
items  = load_item_master()
elapsed = time.time() - t0
print(f"  Both datasets loaded in {elapsed:.1f}s")

check("master shape[0] == 99441",  len(master) == 99441,  str(len(master)))
check("master shape[1] == 42",     master.shape[1] == 42, str(master.shape[1]))
check("items  shape[0] == 112650", len(items)  == 112650, str(len(items)))
check("items  shape[1] == 35",     items.shape[1]  == 35, str(items.shape[1]))
check("master order_id unique",    master["order_id"].nunique() == len(master))
check("items composite unique",    items.groupby(["order_id","order_item_id"]).ngroups == len(items))


# ── 5. No accidental merge / shared columns that imply a join ────────────────
print("\n=== 5. Two-grain architecture — no accidental merge ===")
# Confirm the two DataFrames share ONLY order_id as the documented join key
# and that neither contains a column that would only exist after a merge
# (e.g., master should NOT have item-level 'price' or 'freight_value' columns
#  and items should NOT have order-level 'total_price' or 'max_installments')

master_has_item_cols = any(c in master.columns for c in ["price", "freight_value", "order_item_id"])
items_has_order_cols = any(c in items.columns for c in ["total_price", "max_installments", "primary_payment_type"])

check("master does NOT contain item-grain columns (price/freight_value/order_item_id)",
      not master_has_item_cols,
      str([c for c in ["price","freight_value","order_item_id"] if c in master.columns]))
check("items does NOT contain order-grain-only columns (total_price/max_installments/primary_payment_type)",
      not items_has_order_cols,
      str([c for c in ["total_price","max_installments","primary_payment_type"] if c in items.columns]))


# ── 6. FilterState + apply functions work against both DataFrames ─────────────
print("\n=== 6. Filter pipeline end-to-end ===")
fs = FilterState()   # defaults: Jan2017–Oct2018, delivered only

filtered_m = apply_master_filters(master, fs)
filtered_i = apply_item_filters(items, fs)

check("apply_master_filters returns non-empty DataFrame",  len(filtered_m) > 0)
check("apply_item_filters  returns non-empty DataFrame",   len(filtered_i) > 0)
check("filtered master smaller than original",             len(filtered_m) < len(master),
      f"{len(filtered_m)} vs {len(master)}")
check("filtered items  smaller than original",             len(filtered_i) < len(items),
      f"{len(filtered_i)} vs {len(items)}")
check("filtered master: only delivered",
      (filtered_m["order_status"] == "delivered").all())
check("filtered items:  only delivered",
      (filtered_i["order_status"] == "delivered").all())
check("filtered master: no 2016 partial",
      filtered_m["is_2016_partial"].sum() == 0)
print(f"  Master after default filter: {len(filtered_m):,} rows")
print(f"  Items  after default filter: {len(filtered_i):,} rows")


# ── 7. dataset_summary contract ───────────────────────────────────────────────
print("\n=== 7. dataset_summary() contract ===")
s = dataset_summary()
check("master_exists == True",       s["master_exists"] is True)
check("item_master_exists == True",  s["item_master_exists"] is True)
check("analysis_start == '2017-01-01'", s["analysis_start"] == "2017-01-01")
check("analysis_end   == '2018-10-31'", s["analysis_end"]   == "2018-10-31")


# ── 8. Project tree completeness ─────────────────────────────────────────────
print("\n=== 8. Project tree completeness ===")
required_files = [
    "app.py",
    "requirements.txt",
    "src/__init__.py",
    "src/data_loader.py",
    "src/filters.py",
    "data/olist_master_cleaned.csv",
    "data/olist_order_item_master.csv",
]
for f in required_files:
    exists = Path(f).exists()
    check(f"exists: {f}", exists)


# ── 9. requirements.txt contains all needed packages ─────────────────────────
print("\n=== 9. requirements.txt packages ===")
req_text = Path("requirements.txt").read_text(encoding="utf-8").lower()
for pkg in ["streamlit", "pandas", "numpy", "plotly", "scikit-learn"]:
    check(f"requirements.txt contains '{pkg}'", pkg in req_text)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED — review output above")
    sys.exit(1)
