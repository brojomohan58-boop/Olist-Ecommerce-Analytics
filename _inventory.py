"""Quick inventory script for Phase 2 planning."""
import sys
sys.path.insert(0, ".")
import streamlit as st

def passthrough_cache(*args, **kwargs):
    if args and callable(args[0]): return args[0]
    def decorator(fn): return fn
    return decorator
st.cache_data = passthrough_cache

from src.data_loader import load_master, load_item_master

master = load_master()
items  = load_item_master()

print("=== MASTER COLUMNS + DTYPES + NULL COUNTS ===")
for col in master.columns:
    nulls = master[col].isna().sum()
    print(f"  {col:<45} {str(master[col].dtype):<15} nulls={nulls}")

print()
print("=== ITEM MASTER COLUMNS + DTYPES + NULL COUNTS ===")
for col in items.columns:
    nulls = items[col].isna().sum()
    print(f"  {col:<45} {str(items[col].dtype):<15} nulls={nulls}")

print()
print("=== KEY NUMERIC SUMMARIES (MASTER) ===")
for col in ["total_price","total_freight","total_payment_value","n_items",
            "delivery_time_days","delay_days","review_score","max_installments",
            "avg_item_price","n_payment_methods"]:
    s = master[col].dropna()
    print(f"  {col:<30} count={len(s):>6}  min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}  median={s.median():.2f}")

print()
print("=== KEY NUMERIC SUMMARIES (ITEMS) ===")
for col in ["price","freight_value","delivery_time_days","delay_days","review_score"]:
    s = items[col].dropna()
    print(f"  {col:<30} count={len(s):>6}  min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}  median={s.median():.2f}")

print()
print("=== CARDINALITY CHECK ===")
print(f"  customer_state  (master) : {master['customer_state'].nunique()}")
print(f"  seller_state    (items)  : {items['seller_state'].nunique()}")
print(f"  category        (items)  : {items['product_category_name_english'].nunique()}")
print(f"  seller_id       (items)  : {items['seller_id'].nunique()}")
print(f"  product_id      (items)  : {items['product_id'].nunique()}")
print(f"  customer_unique (master) : {master['customer_unique_id'].nunique()}")
print(f"  order_month_str (master) : {master['order_month_str'].nunique()} months")
print(f"  payment types   (master) : {master['primary_payment_type'].value_counts().to_dict()}")
print(f"  order_status    (master) : {master['order_status'].value_counts().to_dict()}")

print()
print("=== REVIEW SCORE DISTRIBUTION ===")
rv = master["review_score"].value_counts().sort_index()
total_reviews = rv.sum()
for score, cnt in rv.items():
    print(f"  {score:.0f} stars: {cnt:>6}  ({cnt/total_reviews*100:.1f}%)")
print(f"  No review: {master['review_score'].isna().sum()}")

print()
print("=== DELIVERY/DELAY BREAKDOWN (MASTER, DELIVERED ONLY) ===")
delivered = master[master["order_status"] == "delivered"].copy()
print(f"  Delivered orders            : {len(delivered):,}")
print(f"  With delivery date          : {delivered['order_delivered_customer_date'].notna().sum():,}")
delayed_known = delivered[delivered["is_delayed"].notna()]
print(f"  With is_delayed resolved    : {len(delayed_known):,}")
late = delayed_known[delayed_known["is_delayed"] == 1]
ontime = delayed_known[delayed_known["is_delayed"] == 0]
print(f"  On-time                     : {len(ontime):,}  ({len(ontime)/len(delayed_known)*100:.1f}%)")
print(f"  Late                        : {len(late):,}  ({len(late)/len(delayed_known)*100:.1f}%)")
late_days = late["delay_days"].dropna()
print(f"  Avg delay days (late only)  : {late_days.mean():.1f}")
print(f"  Median delivery time (days) : {delivered['delivery_time_days'].median():.1f}")
print(f"  Mean delivery time (days)   : {delivered['delivery_time_days'].mean():.1f}")

print()
print("=== PAYMENT BREAKDOWN ===")
credit = master[master["primary_payment_type"] == "credit_card"]
print(f"  Credit card orders          : {len(credit):,}")
print(f"  Max installments dist (top):")
ic = master["max_installments"].value_counts().sort_index().head(12)
for v, c in ic.items():
    print(f"    {v:.0f}: {c:>6}")

print()
print("=== REPEAT CUSTOMER CHECK ===")
cust_orders = master.groupby("customer_unique_id")["order_id"].count()
repeat = (cust_orders > 1).sum()
single = (cust_orders == 1).sum()
print(f"  Total unique customers : {len(cust_orders):,}")
print(f"  Single-order customers : {single:,}  ({single/len(cust_orders)*100:.1f}%)")
print(f"  Repeat customers (2+)  : {repeat:,}  ({repeat/len(cust_orders)*100:.1f}%)")
print(f"  Max orders per customer: {cust_orders.max()}")
