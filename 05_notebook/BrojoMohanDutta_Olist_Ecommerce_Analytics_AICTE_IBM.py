#!/usr/bin/env python
# coding: utf-8

# # AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
# 
# ## Olist E-Commerce Business Intelligence & Customer Experience Analytics
# 
# **Student:** Brojo Mohan Dutta  
# **Project:** Olist E-Commerce Analytics Platform  
# **Dataset:** Olist Brazilian E-Commerce Public Dataset  
# **Technology:** Python, Pandas, NumPy, Plotly, Scikit-learn
# 
# ### Submission purpose
# This notebook consolidates the project's analytical workflow into a single Jupyter Notebook suitable for the AICTE | IBM SkillsBuild project-code submission.
# 
# The project preserves two analytical grains:
# 
# - **Order level:** one row per order — used for orders, sales, customers, delivery, reviews, payments and regional analysis.
# - **Order-item level:** one row per order item — used for products, categories, sellers, item-level sales and freight analysis.
# 
# The two datasets are intentionally not permanently flattened together.

# In[ ]:


# ============================================================
# 1. Environment & imports
# ============================================================
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

print("Python analytics environment loaded.")


# ## 2. Data loading
# 
# The original project stores the two validated analytical CSVs under `data/`.
# 
# This submission notebook also searches the notebook's working directory so that the code remains convenient to run after downloading the project data.

# In[ ]:


# ============================================================
# 2. Locate and load the two analytical datasets
# ============================================================
ROOT = Path.cwd()

MASTER_CANDIDATES = [
    ROOT / "data" / "olist_master_cleaned.csv",
    ROOT / "olist_master_cleaned.csv",
]
ITEM_CANDIDATES = [
    ROOT / "data" / "olist_order_item_master.csv",
    ROOT / "olist_order_item_master.csv",
]

def first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

MASTER_PATH = first_existing(MASTER_CANDIDATES)
ITEM_PATH = first_existing(ITEM_CANDIDATES)

if MASTER_PATH is None or ITEM_PATH is None:
    raise FileNotFoundError(
        "Required analytical datasets were not found. "
        "Place 'olist_master_cleaned.csv' and "
        "'olist_order_item_master.csv' in a 'data' folder "
        "or beside this notebook."
    )

master = pd.read_csv(MASTER_PATH, low_memory=False)
items = pd.read_csv(ITEM_PATH, low_memory=False)

# Parse the same core timestamp fields used by the project.
for col in [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "review_creation_date",
    "review_answer_timestamp",
]:
    if col in master.columns:
        master[col] = pd.to_datetime(master[col], errors="coerce")

for col in [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
]:
    if col in items.columns:
        items[col] = pd.to_datetime(items[col], errors="coerce")

# Derived temporal fields, matching the project data contract.
for df in (master, items):
    ts = df["order_purchase_timestamp"]
    df["order_year"] = ts.dt.year
    df["order_month_num"] = ts.dt.month
    df["order_month_str"] = ts.dt.to_period("M").astype("string")
    df["order_dow"] = ts.dt.dayofweek

print(f"Master dataset : {master.shape}")
print(f"Item dataset   : {items.shape}")


# In[ ]:


# ============================================================
# 3. Data contract and grain validation
# ============================================================
assert "order_id" in master.columns
assert "order_id" in items.columns
assert master["order_id"].nunique() == len(master), "Order-level grain violation."
assert not items.duplicated(["order_id", "order_item_id"]).any(), "Item-level grain violation."

print("ORDER GRAIN  :", "one row per order")
print("ITEM GRAIN   :", "one row per order-item")
print("Master rows  :", f"{len(master):,}")
print("Item rows    :", f"{len(items):,}")
print("Master cols  :", master.shape[1])
print("Item cols    :", items.shape[1])
print("Grain checks : PASS")


# ## 4. Analysis filters
# 
# The project's default analytical period is January 2017 through October 2018, with delivered orders used for the primary operational KPIs. Full-status data is retained for status and cancellation analysis.

# In[ ]:


# ============================================================
# 4. Default analysis slices
# ============================================================
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2018-10-17 23:59:59")

master_period = master[
    (master["order_purchase_timestamp"] >= START) &
    (master["order_purchase_timestamp"] <= END) &
    (master["is_2016_partial"].fillna(False) == False)
].copy()

master_delivered = master_period[
    master_period["order_status"].astype(str).eq("delivered")
].copy()

items_period = items[
    (items["order_purchase_timestamp"] >= START) &
    (items["order_purchase_timestamp"] <= END) &
    (items["is_primary_analysis_period"].fillna(False) == True)
].copy()

print("Primary order rows   :", f"{len(master_period):,}")
print("Delivered orders     :", f"{len(master_delivered):,}")
print("Primary item rows    :", f"{len(items_period):,}")


# ## 5. Executive KPI analysis
# 
# The KPI definitions follow the project's `src/kpis.py` logic: order-level metrics use the order-grain table, while item/category/seller metrics use the order-item grain.

# In[ ]:


# ============================================================
# 5. Executive KPIs
# ============================================================
def safe_sum(s):
    return float(s.sum(skipna=True))

def safe_mean(s):
    s = s.dropna()
    return float(s.mean()) if len(s) else 0.0

def safe_median(s):
    s = s.dropna()
    return float(s.median()) if len(s) else 0.0

def rate(num, den):
    return float(num / den) if den else 0.0

total_orders = len(master_period)
total_sales = safe_sum(master_period["total_price"])
total_freight = safe_sum(master_period["total_freight"])
aov = safe_mean(master_period["total_price"])
unique_customers = master_period["customer_unique_id"].nunique()

delivery_days = master_delivered["delivery_time_days"].dropna()
avg_delivery = safe_mean(delivery_days)
median_delivery = safe_median(delivery_days)

delivered_known = master_delivered[master_delivered["is_delayed"].notna()]
delay_rate = rate(int(delivered_known["is_delayed"].eq(1).sum()), len(delivered_known))
on_time_rate = rate(int(delivered_known["is_delayed"].eq(0).sum()), len(delivered_known))

reviewed = master_period["review_score"].dropna()
avg_review = safe_mean(reviewed)
five_star_rate = rate(int(reviewed.eq(5).sum()), len(reviewed))
low_score_rate = rate(int(reviewed.isin([1, 2]).sum()), len(reviewed))
review_coverage = rate(len(reviewed), len(master_period))

repeat_counts = master_period.groupby("customer_unique_id")["order_id"].nunique()
repeat_customers = int((repeat_counts >= 2).sum())
repeat_rate = rate(repeat_customers, len(repeat_counts))

cancelled = master_period["order_status"].astype(str).eq("canceled").sum()
cancellation_rate = rate(int(cancelled), len(master_period))

kpi_table = pd.DataFrame({
    "Metric": [
        "Total Orders", "Total Sales Value (R$)", "Total Freight (R$)",
        "Average Order Value (R$)", "Unique Customers",
        "Average Delivery Days", "Median Delivery Days",
        "On-Time Rate", "Delay Rate", "Average Review Score",
        "5-Star Rate", "Low-Score Rate", "Review Coverage",
        "Repeat Customer Rate", "Cancellation Rate"
    ],
    "Value": [
        total_orders, total_sales, total_freight, aov, unique_customers,
        avg_delivery, median_delivery, on_time_rate, delay_rate,
        avg_review, five_star_rate, low_score_rate, review_coverage,
        repeat_rate, cancellation_rate
    ]
})

display(kpi_table)


# ## 6. Sales & order performance

# In[ ]:


# ============================================================
# 6. Sales and order trends
# ============================================================
monthly = (
    master_period
    .groupby("order_month_str", as_index=False)
    .agg(
        order_count=("order_id", "count"),
        total_sales=("total_price", "sum"),
        avg_order_value=("total_price", "mean"),
    )
    .sort_values("order_month_str")
)

fig = px.line(
    monthly,
    x="order_month_str",
    y="total_sales",
    markers=True,
    title="Monthly Sales Value",
    labels={"order_month_str": "Month", "total_sales": "Sales Value (R$)"}
)
fig.update_layout(template="plotly_white")
fig.show()

fig = px.bar(
    monthly,
    x="order_month_str",
    y="order_count",
    title="Monthly Order Volume",
    labels={"order_month_str": "Month", "order_count": "Orders"}
)
fig.update_layout(template="plotly_white")
fig.show()

display(monthly.sort_values("total_sales", ascending=False).head(10))


# ## 7. Product and category performance

# In[ ]:


# ============================================================
# 7. Product / category analysis
# ============================================================
category = (
    items_period
    .groupby("product_category_name_english", dropna=False)
    .agg(
        item_count=("order_item_id", "count"),
        unique_orders=("order_id", "nunique"),
        unique_sellers=("seller_id", "nunique"),
        total_sales_value=("sales_value", "sum"),
        avg_item_price=("price", "mean"),
        total_freight_value=("freight_value", "sum"),
    )
    .reset_index()
    .rename(columns={"product_category_name_english": "category"})
)

category["category"] = category["category"].fillna("Unknown")

top_categories = category.sort_values("total_sales_value", ascending=False).head(15)

fig = px.bar(
    top_categories.sort_values("total_sales_value"),
    x="total_sales_value",
    y="category",
    orientation="h",
    title="Top Product Categories by Sales Value",
    labels={"total_sales_value": "Sales Value (R$)", "category": "Category"}
)
fig.update_layout(template="plotly_white")
fig.show()

display(top_categories)


# ## 8. Delivery and operational performance

# In[ ]:


# ============================================================
# 8. Delivery analysis
# ============================================================
delivery_by_month = (
    master_delivered
    .groupby("order_month_str", as_index=False)
    .agg(
        orders=("order_id", "count"),
        avg_delivery_days=("delivery_time_days", "mean"),
        delay_rate=("is_delayed", "mean"),
    )
    .sort_values("order_month_str")
)

delivery_by_month["delay_rate_pct"] = delivery_by_month["delay_rate"] * 100

fig = px.line(
    delivery_by_month,
    x="order_month_str",
    y="avg_delivery_days",
    markers=True,
    title="Average Delivery Time by Month",
    labels={"order_month_str": "Month", "avg_delivery_days": "Days"}
)
fig.update_layout(template="plotly_white")
fig.show()

fig = px.line(
    delivery_by_month,
    x="order_month_str",
    y="delay_rate_pct",
    markers=True,
    title="Delivery Delay Rate by Month",
    labels={"order_month_str": "Month", "delay_rate_pct": "Delay Rate (%)"}
)
fig.update_layout(template="plotly_white")
fig.show()

state_delivery = (
    master_delivered
    .groupby("customer_state", as_index=False)
    .agg(
        order_count=("order_id", "count"),
        avg_delivery_days=("delivery_time_days", "mean"),
        delay_rate=("is_delayed", "mean"),
    )
)
state_delivery["delay_rate_pct"] = state_delivery["delay_rate"] * 100
display(state_delivery.sort_values("delay_rate", ascending=False).head(10))


# ## 9. Customer experience and reviews

# In[ ]:


# ============================================================
# 9. Customer experience
# ============================================================
review_distribution = (
    master_period["review_score"]
    .value_counts()
    .sort_index()
    .rename_axis("review_score")
    .reset_index(name="count")
)
review_distribution["share_pct"] = (
    review_distribution["count"] / review_distribution["count"].sum() * 100
)

fig = px.bar(
    review_distribution,
    x="review_score",
    y="count",
    text="share_pct",
    title="Review Score Distribution",
    labels={"review_score": "Review Score", "count": "Orders with Review"}
)
fig.update_layout(template="plotly_white")
fig.show()

review_delay = (
    master_delivered
    .dropna(subset=["review_score", "is_delayed"])
    .groupby("is_delayed", as_index=False)
    .agg(
        avg_review_score=("review_score", "mean"),
        order_count=("order_id", "count")
    )
)
review_delay["delivery_status"] = review_delay["is_delayed"].map(
    {0: "On Time", 1: "Delayed"}
)
display(review_delay)


# ## 10. Payment behavior

# In[ ]:


# ============================================================
# 10. Payment analysis
# ============================================================
payment = (
    master_period
    .groupby("primary_payment_type", dropna=False)
    .agg(
        order_count=("order_id", "count"),
        avg_order_value=("total_price", "mean"),
        median_order_value=("total_price", "median"),
    )
    .reset_index()
)

payment["share_pct"] = payment["order_count"] / payment["order_count"].sum() * 100
payment["primary_payment_type"] = payment["primary_payment_type"].astype(str)

fig = px.pie(
    payment,
    names="primary_payment_type",
    values="order_count",
    title="Primary Payment Method Share",
    hole=0.45
)
fig.update_layout(template="plotly_white")
fig.show()

display(payment.sort_values("order_count", ascending=False))


# ## 11. Regional performance

# In[ ]:


# ============================================================
# 11. Regional analysis
# ============================================================
regional = (
    master_period
    .groupby("customer_state", as_index=False)
    .agg(
        order_count=("order_id", "count"),
        total_sales=("total_price", "sum"),
        avg_order_value=("total_price", "mean"),
        unique_customers=("customer_unique_id", "nunique"),
        total_freight=("total_freight", "sum"),
    )
    .sort_values("total_sales", ascending=False)
)

fig = px.bar(
    regional.head(15).sort_values("total_sales"),
    x="total_sales",
    y="customer_state",
    orientation="h",
    title="Top Customer States by Sales",
    labels={"total_sales": "Sales Value (R$)", "customer_state": "State"}
)
fig.update_layout(template="plotly_white")
fig.show()

display(regional.head(10))


# ## 12. Seller performance

# In[ ]:


# ============================================================
# 12. Seller analysis
# ============================================================
seller = (
    items_period
    .groupby("seller_id", as_index=False)
    .agg(
        item_count=("order_item_id", "count"),
        unique_orders=("order_id", "nunique"),
        total_sales=("sales_value", "sum"),
        total_freight=("freight_value", "sum"),
        avg_item_price=("price", "mean"),
        avg_delivery_days=("delivery_time_days", "mean"),
    )
    .sort_values("total_sales", ascending=False)
)

display(seller.head(20))

fig = px.bar(
    seller.head(15).sort_values("total_sales"),
    x="total_sales",
    y="seller_id",
    orientation="h",
    title="Top Sellers by Sales Value",
    labels={"total_sales": "Sales Value (R$)", "seller_id": "Seller ID"}
)
fig.update_layout(template="plotly_white")
fig.show()


# ## 13. Order status and business-quality checks

# In[ ]:


# ============================================================
# 13. Data quality / reconciliation checks
# ============================================================
status_summary = (
    master
    .assign(order_status=master["order_status"].astype(str))
    .groupby("order_status", as_index=False)
    .agg(order_count=("order_id", "count"))
    .sort_values("order_count", ascending=False)
)
status_summary["share_pct"] = status_summary["order_count"] / len(master) * 100

checks = {
    "Master order_id unique": master["order_id"].nunique() == len(master),
    "Item composite key unique": not items.duplicated(["order_id", "order_item_id"]).any(),
    "Master row count = 99,441": len(master) == 99441,
    "Item row count = 112,650": len(items) == 112650,
    "27 customer states": master["customer_state"].nunique() == 27,
    "72 product categories": items["product_category_name_english"].nunique() == 72,
}

quality = pd.DataFrame({
    "Check": list(checks.keys()),
    "Result": ["PASS" if v else "REVIEW" for v in checks.values()]
})
display(quality)
display(status_summary)


# ## 14. Machine-learning component: delivery-delay prediction
# 
# The project's ML module uses a leakage-safe prediction design. The target is `is_delayed`, and post-delivery variables such as actual delivery dates, delivery duration, delay days, and review scores must not be used as predictors.
# 
# If the project's `ml_delay_dataset.csv` is available, the following section trains and evaluates Logistic Regression and Random Forest models using a chronological split.

# In[ ]:


# ============================================================
# 14. ML delay prediction — optional execution
# ============================================================
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ML_CANDIDATES = [
    ROOT / "data" / "ml_delay_dataset.csv",
    ROOT / "ml_delay_dataset.csv",
]
ML_PATH = first_existing(ML_CANDIDATES)

if ML_PATH is None:
    print("ML dataset not found.")
    print("The project ML module expects: data/ml_delay_dataset.csv")
else:
    ml = pd.read_csv(ML_PATH, parse_dates=["order_purchase_timestamp"])
    print("ML dataset shape:", ml.shape)

    forbidden = {
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "delivery_time_days",
        "delay_days",
        "review_score",
        "review_creation_date",
        "review_answer_timestamp",
    }
    leakage = forbidden.intersection(ml.columns)
    if leakage:
        raise ValueError(f"Potential leakage columns detected: {sorted(leakage)}")

    target = "is_delayed"
    timestamp = "order_purchase_timestamp"

    categorical = [
        c for c in [
            "primary_payment_type",
            "product_category_name_english",
            "customer_state",
            "seller_state",
        ] if c in ml.columns
    ]

    X = ml.drop(columns=[target, timestamp], errors="ignore")
    y = ml[target].astype(int)

    # Remove other non-predictive high-cardinality identifiers if present.
    id_like = [
        c for c in X.columns
        if c.endswith("_id") and c not in categorical
    ]
    X = X.drop(columns=id_like, errors="ignore")

    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    # Chronological split: 70% train, 15% validation, 15% test by time.
    ml = ml.sort_values(timestamp).reset_index(drop=True)
    n = len(ml)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = ml.iloc[:train_end]
    valid = ml.iloc[train_end:valid_end]
    test = ml.iloc[valid_end:]

    X_train = train.drop(columns=[target, timestamp], errors="ignore")
    X_valid = valid.drop(columns=[target, timestamp], errors="ignore")
    X_test = test.drop(columns=[target, timestamp], errors="ignore")
    y_train = train[target].astype(int)
    y_valid = valid[target].astype(int)
    y_test = test[target].astype(int)

    for frame in (X_train, X_valid, X_test):
        frame.drop(columns=[c for c in id_like if c in frame.columns], inplace=True)

    logistic = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])

    results = []
    for name, model in [("Logistic Regression", logistic), ("Random Forest", rf)]:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        results.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, proba),
            "Average Precision": average_precision_score(y_test, proba),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
        })

    ml_results = pd.DataFrame(results)
    display(ml_results)


# ## 15. Key analytical outputs
# 
# The final project combines descriptive BI analysis with the delivery-delay ML component. The notebook intentionally reports measurements and analytical observations rather than inventing unsupported business claims.

# In[ ]:


# ============================================================
# 15. Compact analytical summary
# ============================================================
summary = pd.DataFrame({
    "Area": [
        "Orders", "Sales", "Customers", "Delivery",
        "Reviews", "Payments", "Categories", "Sellers"
    ],
    "Key Measure": [
        f"{total_orders:,}",
        f"R$ {total_sales:,.2f}",
        f"{unique_customers:,}",
        f"{on_time_rate*100:.1f}% on-time",
        f"{avg_review:.2f} average score",
        f"{payment.loc[payment['order_count'].idxmax(), 'primary_payment_type']}",
        f"{category['category'].nunique()} categories",
        f"{items_period['seller_id'].nunique():,} sellers",
    ]
})
display(summary)


# ## 16. Reproducibility notes
# 
# ### Required analytical files
# - `olist_master_cleaned.csv`
# - `olist_order_item_master.csv`
# 
# ### Optional ML file
# - `ml_delay_dataset.csv`
# 
# ### Required Python libraries
# - pandas
# - numpy
# - plotly
# - scikit-learn
# 
# The project's original Streamlit application, source modules, page modules, validation scripts, README, and requirements file are maintained separately in the GitHub/project repository. This notebook consolidates the core analytical and ML workflow into the single code artifact required for the internship submission.

# # Appendix — Project Source Code Archive
# 
# The following cells preserve the uploaded project source modules in the submission notebook. They are included so the notebook contains the project's source-code reference in addition to the consolidated analytical workflow above.

# ### `app.py` — Application entry point

# In[ ]:


"""
app.py
======
Entry point for the Olist E-Commerce Analytics Platform.

Architecture
------------
Streamlit multi-page app using st.navigation() + st.Page().
All pages are defined as callables here during Phase 1.
In Phase 2+ each page is replaced by a dedicated pages/*.py file.

Data contract
-------------
- load_master()      → order-level DataFrame (99 441 rows × 42 cols)
- load_item_master() → item-level DataFrame  (112 650 rows × 35 cols)
- These two DataFrames are NEVER permanently merged or flattened.
- render_sidebar_filters() returns a FilterState; pages call
  apply_master_filters() / apply_item_filters() independently.
"""

import streamlit as st

from src.data_loader import load_master, load_item_master, dataset_summary
from src.filters import render_sidebar_filters

# ---------------------------------------------------------------------------
# Global page configuration  (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Olist E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Olist E-Commerce Analytics Platform**\n\n"
            "Built on the Olist Brazilian E-Commerce Public Dataset.\n"
            "Analysis period: January 2017 – October 2018."
        ),
    },
)

# ---------------------------------------------------------------------------
# Load data (cached — runs once per session)
# ---------------------------------------------------------------------------

master = load_master()
items  = load_item_master()

# ---------------------------------------------------------------------------
# Page callables (Phase 1 stubs — replaced by pages/*.py in Phase 2+)
# ---------------------------------------------------------------------------





def page_sellers() -> None:
    fs = render_sidebar_filters(master, items)

    st.title("🏪 Seller Intelligence")
    st.caption(
        "Top sellers by revenue, delivery performance, "
        "and freight competitiveness."
    )
    st.info(
        "This page will be implemented in Phase 3 (P3-2).",
        icon="ℹ️",
    )


def page_ml() -> None:
    import importlib.util
    from pathlib import Path

    ml_page_path = (
        Path(__file__).parent
        / "pages"
        / "08_ml_delay_predictor.py"
    )

    spec = importlib.util.spec_from_file_location(
        "ml_delay_predictor_page",
        ml_page_path,
    )

    if spec is None or spec.loader is None:
        st.error("Unable to load the ML Delay Predictor page.")
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def page_ml() -> None:
    fs = render_sidebar_filters(master, items)
    st.title("🤖 ML: Delivery Delay Predictor")
    st.caption("Order-level delay prediction model — feature importance, evaluation metrics.")
    st.info("🚧 This page will be implemented in Phase 4 (P4-2).", icon="ℹ️")


def page_about() -> None:
    """Static data-health and project info page — no filters needed."""
    st.title("ℹ️ About this App")
    st.markdown(
        """
        **Olist E-Commerce Analytics Platform**

        Built on the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

        ---
        ### Data Architecture
        | Dataset | Grain | Rows | Columns |
        |---|---|---|---|
        | `olist_master_cleaned.csv` | One row per **order** | 99,441 | 38 source + 4 derived |
        | `olist_order_item_master.csv` | One row per **order-item** | 112,650 | 32 source + 3 derived |

        > The two datasets are **never permanently merged**. Each analysis page
        > queries the appropriate dataset independently to prevent double-counting.

        ---
        ### Analysis Period
        Default: **January 2017 – October 2018**

        The 329 orders from Sep–Dec 2016 represent a partial year and are
        excluded by default. Use the sidebar toggle to include them.

        ---
        ### Key Business Constraints
        - The dataset does **not** contain product cost / COGS — profitability
          and margin metrics are not computed.
        - `delay_days` is signed: **negative values mean the order arrived early**.
          "Average delay" KPIs show only late orders (`delay_days > 0`).
        - Payment-vs-sales differences are shown as a reconciliation metric only
          — no causal attribution is made.
        """
    )

    st.divider()
    st.subheader("Dataset health check")

    summary = dataset_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Master CSV found",      "✅ Yes" if summary["master_exists"]      else "❌ Missing")
    col2.metric("Item Master CSV found", "✅ Yes" if summary["item_master_exists"] else "❌ Missing")
    col3.metric("Analysis window", f"{summary['analysis_start']}  →  {summary['analysis_end']}")

    st.divider()
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Master — loaded shape")
        st.write(f"**{len(master):,} rows × {master.shape[1]} columns**")
        st.write(f"Unique orders: **{master['order_id'].nunique():,}**")
        st.write(f"Unique customers: **{master['customer_unique_id'].nunique():,}**")
        yr = master["order_year"].value_counts().sort_index()
        st.write("Orders by year:")
        st.dataframe(
            yr.rename("orders").reset_index().rename(columns={"order_year": "year"}),
            hide_index=True,
            use_container_width=False,
        )
    with col5:
        st.subheader("Item Master — loaded shape")
        st.write(f"**{len(items):,} rows × {items.shape[1]} columns**")
        st.write(f"Unique orders: **{items['order_id'].nunique():,}**")
        st.write(f"Unique products: **{items['product_id'].nunique():,}**")
        st.write(f"Unique sellers: **{items['seller_id'].nunique():,}**")
        st.write(f"Unique categories: **{items['product_category_name_english'].nunique():,}**")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

_pages = st.navigation(
    {
        "Analytics": [
            st.Page("pages/01_overview.py", title="Executive Overview", icon="📊", default=True),
            st.Page("pages/02_products_categories.py", title="Products & Categories", icon="📦"),
            st.Page("pages/03_delivery_operations.py", title="Delivery & Operations", icon="🚚"),
            st.Page("pages/04_customer_experience.py", title="Customer Experience", icon="⭐"),
            st.Page("pages/05_payments.py", title="Payments", icon="💳"),
            st.Page("pages/06_regional.py", title="Regional Analysis", icon="🗺️"),
            st.Page("pages/07_sellers.py", title="Seller Intelligence", icon="🏪"),
        ],
        "Machine Learning": [
            st.Page("pages/08_ml_delay_predictor.py", title="Delay Predictor", icon="🤖"),
        ],
        "Info": [
            st.Page(page_about,    title="About",                   icon="ℹ️"),
        ],
    }
)

_pages.run()


# ### `data_loader(2).py` — Data loading and analytical-grain logic

# In[ ]:


"""
data_loader.py
==============
Cached loaders for the two source datasets.  The two DataFrames are NEVER
merged into a permanent flat table — each preserves its own grain.

Grains
------
- Master        : one row per order  (order_id is unique)
- Item Master   : one row per order-item  (order_id + order_item_id composite)

Key helpers exposed
-------------------
load_master()
    → DataFrame, 99 441 rows, 38 cols + derived columns
    Grain assertion: order_id is unique.

load_item_master()
    → DataFrame, 112 650 rows, 32 cols + derived columns
    Grain assertion: (order_id, order_item_id) is unique.

build_order_category_table(item_df)
    → DataFrame, one row per order.
    Primary category = mode of product_category_name_english for that order.
    Deterministic tie-break: alphabetically first category wins.
    Used for category-level delivery and review metrics only.
    Documents multi-category orders via the `has_multiple_categories` flag.

DATA PATHS
----------
Both CSVs are expected in  data/  relative to the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent  # project root
_MASTER_PATH = _ROOT / "data" / "olist_master_cleaned.csv"
_ITEM_PATH = _ROOT / "data" / "olist_order_item_master.csv"

# ---------------------------------------------------------------------------
# Analysis period constants
# ---------------------------------------------------------------------------

ANALYSIS_START = pd.Timestamp("2017-01-01")
ANALYSIS_END = pd.Timestamp("2018-10-31")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_path(path: Path) -> None:
    """Raise FileNotFoundError with a clear message if the CSV is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place both CSVs in the data/ directory before running the app."
        )


def _parse_timestamps(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Parse a list of columns to datetime, coercing unparseable values to NaT."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _safe_int_flag(series: pd.Series) -> pd.Series:
    """
    Convert a 0.0 / 1.0 float flag column (with possible NaN) to
    Int8 nullable integer so that:
      - 0.0  → 0
      - 1.0  → 1
      - NaN  → pd.NA   (not delivered / unknown)
    Using Int8 (capital I) preserves pd.NA without forcing float.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int8")


# ---------------------------------------------------------------------------
# Master loader  (order-grain)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading order master…")
def load_master() -> pd.DataFrame:
    """
    Load olist_master_cleaned.csv.

    Grain: one row per order (order_id is unique).

    Post-load transformations
    -------------------------
    Datetime columns parsed:
      order_purchase_timestamp, order_approved_at,
      order_delivered_carrier_date, order_delivered_customer_date,
      order_estimated_delivery_date, review_creation_date,
      review_answer_timestamp

    Derived columns added:
      order_year        int16   — year of purchase
      order_month_num   int8    — month of purchase (1-12)
      order_month_str   str     — "YYYY-MM" label for trend charts
      order_dow         int8    — day of week (0=Mon … 6=Sun)
      is_delayed        Int8    — 0 / 1 / pd.NA (nullable int; NA = not delivered)
      is_2016_partial   bool    — True for the 329 partial-year 2016 orders

    Assertions
    ----------
    Raises AssertionError if order_id is not unique.
    """
    _validate_path(_MASTER_PATH)

    df = pd.read_csv(
        _MASTER_PATH,
        dtype={
            "order_id": "string",
            "customer_id": "string",
            "customer_unique_id": "string",
            "order_status": "category",
            "customer_state": "category",
            "seller_state": "category",
            "primary_payment_type": "category",
            "product_category_name_english": "category",
            "seller_city": "string",
            "customer_city": "string",
        },
        low_memory=False,
    )

    # --- datetime parsing ---
    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_creation_date",
        "review_answer_timestamp",
    ]
    df = _parse_timestamps(df, timestamp_cols)

    # --- flag columns ---
    df["is_delayed"] = _safe_int_flag(df["is_delayed"])
    df["is_2016_partial"] = df["is_2016_partial"].map(
        {"True": True, "False": False, True: True, False: False}
    ).fillna(False).astype(bool)

    # --- derived temporal columns ---
    ts = df["order_purchase_timestamp"]
    df["order_year"] = ts.dt.year.astype("Int16")
    df["order_month_num"] = ts.dt.month.astype("Int8")
    df["order_month_str"] = ts.dt.to_period("M").astype("string")
    df["order_dow"] = ts.dt.dayofweek.astype("Int8")

    # --- numeric coercions ---
    for col in ["total_price", "total_freight", "total_payment_value",
                "avg_item_price", "n_items", "delivery_time_days",
                "delay_days", "review_score", "max_installments",
                "n_payment_methods"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- grain assertion ---
    assert df["order_id"].nunique() == len(df), (
        f"GRAIN VIOLATION: olist_master_cleaned.csv — "
        f"order_id is not unique ({df['order_id'].nunique()} unique IDs "
        f"in {len(df)} rows)."
    )

    return df


# ---------------------------------------------------------------------------
# Item Master loader  (order-item grain)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading item master…")
def load_item_master() -> pd.DataFrame:
    """
    Load olist_order_item_master.csv.

    Grain: one row per order-item  (order_id, order_item_id composite key).

    Post-load transformations
    -------------------------
    Datetime columns parsed:
      order_purchase_timestamp, order_delivered_customer_date,
      order_estimated_delivery_date, shipping_limit_date

    Derived columns added:
      order_year        int16
      order_month_num   int8
      order_month_str   str   — "YYYY-MM"
      order_dow         int8
      is_delayed        Int8  — 0 / 1 / pd.NA
      is_primary_analysis_period  bool

    Assertions
    ----------
    Raises AssertionError if (order_id, order_item_id) composite is not unique.
    """
    _validate_path(_ITEM_PATH)

    df = pd.read_csv(
        _ITEM_PATH,
        dtype={
            "order_id": "string",
            "order_item_id": "string",
            "product_id": "string",
            "seller_id": "string",
            "order_status": "category",
            "customer_state": "category",
            "seller_state": "category",
            "product_category_name_english": "category",
            "seller_city": "string",
            "customer_unique_id": "string",
            "customer_id": "string",
        },
        low_memory=False,
    )

    # --- datetime parsing ---
    timestamp_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",
    ]
    df = _parse_timestamps(df, timestamp_cols)

    # --- flag columns ---
    df["is_delayed"] = _safe_int_flag(df["is_delayed"])
    df["is_primary_analysis_period"] = df["is_primary_analysis_period"].map(
        {"True": True, "False": False, True: True, False: False}
    ).fillna(False).astype(bool)

    # --- derived temporal columns ---
    ts = df["order_purchase_timestamp"]
    df["order_year"] = ts.dt.year.astype("Int16")
    df["order_month_num"] = ts.dt.month.astype("Int8")
    df["order_month_str"] = ts.dt.to_period("M").astype("string")
    df["order_dow"] = ts.dt.dayofweek.astype("Int8")

    # --- numeric coercions ---
    for col in ["price", "freight_value", "delivery_time_days", "delay_days",
                "review_score", "total_payment_value", "sales_value",
                "item_total_value", "product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- grain assertion ---
    composite_unique = df.groupby(["order_id", "order_item_id"]).ngroups
    assert composite_unique == len(df), (
        f"GRAIN VIOLATION: olist_order_item_master.csv — "
        f"(order_id, order_item_id) composite is not unique "
        f"({composite_unique} unique combos in {len(df)} rows)."
    )

    return df


# ---------------------------------------------------------------------------
# Order-category attribution table
# ---------------------------------------------------------------------------

def build_order_category_table(item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an order-level category attribution table with exactly one row
    per order.

    Algorithm
    ---------
    For each order, the primary_category is the mode of
    product_category_name_english across all items.

    Tie-breaking (deterministic)
    ----------------------------
    When two or more categories appear equally often in the same order,
    the alphabetically first category name is selected.  This guarantees
    identical results on every run regardless of row ordering.

    Output columns
    --------------
    order_id                   string
    primary_category           string   — most-frequent category for the order
    n_categories               int8     — distinct categories in the order
    has_multiple_categories    bool     — True when n_categories > 1
    is_delayed                 Int8     — copied from item (same value per order)
    delivery_time_days         float    — copied from item (same value per order)
    delay_days                 float    — copied from item (same value per order)
    review_score               float    — copied from item (same value per order)
    customer_state             category — copied from item
    order_purchase_timestamp   datetime — copied from item
    order_month_str            string   — "YYYY-MM"
    is_primary_analysis_period bool

    Notes
    -----
    - This table is built transiently at runtime and is never persisted.
    - Delivery / review fields are order-level values that are identical
      across all items of the same order in the source data.  We take the
      first value after sorting by order_item_id to ensure determinism.
    - Multi-category attribution is documented via has_multiple_categories.
    """
    cat_col = "product_category_name_english"

    # -- 1. Compute primary_category with deterministic tie-break -------------
    # Vectorised approach — no per-group Python callback.
    #
    # Algorithm:
    #   a) Count occurrences of each (order_id, category) pair → item_count.
    #   b) Sort by (order_id ASC, item_count DESC, category ASC).
    #      - item_count DESC  → most-frequent category rises to the top.
    #      - category ASC     → alphabetically first name wins any count tie.
    #   c) drop_duplicates(order_id, keep='first') picks the winner per order.
    #
    # This is fully C-level in pandas and runs in < 1 second on 112 650 rows.

    clean_df = item_df.dropna(subset=[cat_col]).copy()
    clean_df[cat_col] = clean_df[cat_col].astype(str)

    counts_df = (
        clean_df
        .groupby(["order_id", cat_col], sort=False)
        .size()
        .reset_index(name="item_count")
    )

    counts_df = counts_df.sort_values(
        ["order_id", "item_count", cat_col],
        ascending=[True, False, True],  # count DESC, category ASC for tie-break
    )

    category_map = (
        counts_df
        .drop_duplicates(subset=["order_id"], keep="first")
        [["order_id", cat_col]]
        .rename(columns={cat_col: "primary_category"})
        .reset_index(drop=True)
    )

    # -- 2. Count distinct categories per order --------------------------------
    n_cat = (
        clean_df
        .groupby("order_id", sort=False)[cat_col]
        .nunique()
        .rename("n_categories")
        .reset_index()
    )

    # -- 3. Order-level fields (take first row per order after sort by item_id)
    order_level_cols = [
        "order_id",
        "is_delayed",
        "delivery_time_days",
        "delay_days",
        "review_score",
        "customer_state",
        "order_purchase_timestamp",
        "order_month_str",
        "is_primary_analysis_period",
    ]
    available_cols = [c for c in order_level_cols if c in item_df.columns]

    first_per_order = (
        item_df
        .sort_values(["order_id", "order_item_id"])
        .drop_duplicates(subset=["order_id"], keep="first")
        [available_cols]
    )

    # -- 4. Assemble ----------------------------------------------------------
    result = (
        category_map
        .merge(n_cat, on="order_id", how="left")
        .merge(first_per_order, on="order_id", how="left")
    )
    result["n_categories"] = result["n_categories"].fillna(1).astype("Int8")
    result["has_multiple_categories"] = result["n_categories"] > 1

    # -- 5. Grain assertion ---------------------------------------------------
    assert result["order_id"].nunique() == len(result), (
        "GRAIN VIOLATION: build_order_category_table produced duplicate order_ids."
    )

    return result


# ---------------------------------------------------------------------------
# Convenience summary (for debugging / startup validation)
# ---------------------------------------------------------------------------

def dataset_summary() -> dict:
    """
    Return a lightweight summary dict — useful for the app's About page
    or startup health-check.  Does NOT re-load cached data.
    """
    return {
        "master_path": str(_MASTER_PATH),
        "item_master_path": str(_ITEM_PATH),
        "master_exists": _MASTER_PATH.exists(),
        "item_master_exists": _ITEM_PATH.exists(),
        "analysis_start": str(ANALYSIS_START.date()),
        "analysis_end": str(ANALYSIS_END.date()),
    }


# ### `filters(1).py` — Filtering logic

# In[ ]:


"""
filters.py
==========
Shared sidebar filter widgets for the Olist analytics application.

Design principles
-----------------
- One function, `render_sidebar_filters()`, renders all controls and returns
  a `FilterState` dataclass whose fields are plain Python values (dates,
  lists of strings, booleans).
- Two apply functions — `apply_master_filters()` and
  `apply_item_filters()` — accept the FilterState and their respective
  DataFrames and return filtered copies. They never mutate the originals.
- The two DataFrames are always filtered independently; they are never
  joined inside this module (two-grain architecture preserved).
- All Streamlit widgets are placed in st.sidebar so every page inherits
  the same controls without any additional setup.

Default analysis period
-----------------------
January 2017 - October 2018 (2016 partial-period orders excluded by default).

2016 handling
-------------
`order_purchase_timestamp` is the SOLE authority for the 2016 cutoff.
When include_2016=False, rows with order_purchase_timestamp < 2017-01-01
are excluded via a direct timestamp comparison — this does NOT depend on
any helper flag column (is_2016_partial / is_primary_analysis_period)
existing in the DataFrame. This is a deliberate fix: relying on a flag
column that may be absent or stale silently disabled the toggle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Analysis-period boundaries (must match data_loader.py constants)
# ---------------------------------------------------------------------------

_ANALYSIS_START = date(2017, 1, 1)
_ANALYSIS_END   = date(2018, 10, 17)   # last observed order date in the dataset
_FULL_START     = date(2016, 9, 4)     # earliest record (partial-year 2016)

_ANALYSIS_START_TS = pd.Timestamp(_ANALYSIS_START)

_ALL_STATUSES = [
    "delivered",
    "shipped",
    "invoiced",
    "processing",
    "canceled",
    "unavailable",
    "approved",
    "created",
]

_RESET_KEYS = [
    "filter_date_start", "filter_date_end", "filter_include_2016",
    "filter_order_statuses", "filter_customer_states",
    "filter_seller_states", "filter_categories",
]


def _format_status_label(value: str) -> str:
    """Human-readable order-status label while preserving the raw value."""
    return str(value).replace("_", " ").strip().title()


def _format_category_label(value: str) -> str:
    """Human-readable category label while preserving the raw value."""
    return str(value).replace("_", " ").strip().title()


def _reset_filters() -> None:
    """Reset every sidebar widget to its actual default state."""
    st.session_state["filter_date_start"] = _ANALYSIS_START
    st.session_state["filter_date_end"] = _ANALYSIS_END
    st.session_state["filter_include_2016"] = False
    st.session_state["filter_order_statuses"] = ["delivered"]
    st.session_state["filter_customer_states"] = []
    st.session_state["filter_seller_states"] = []
    st.session_state["filter_categories"] = []


# ---------------------------------------------------------------------------
# FilterState dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterState:
    """
    Plain-data container for all active filter values.

    Attributes
    ----------
    date_start : date
        Inclusive lower bound on order_purchase_timestamp.
    date_end : date
        Inclusive upper bound on order_purchase_timestamp.
    include_2016 : bool
        When False (default), orders with order_purchase_timestamp before
        2017-01-01 are excluded from BOTH the Master and Item DataFrames,
        based directly on the timestamp (not a flag column).
    order_statuses : list[str]
        Allowed values of order_status. Empty list = no status filter.
    customer_states : list[str]
        Allowed customer_state values. Empty list = all states.
    seller_states : list[str]
        Allowed seller_state values. Empty list = all states.
    categories : list[str]
        Allowed product_category_name_english values. Empty list = all.
    """

    date_start:       date       = field(default_factory=lambda: _ANALYSIS_START)
    date_end:         date       = field(default_factory=lambda: _ANALYSIS_END)
    include_2016:     bool       = False
    order_statuses:   list[str]  = field(default_factory=lambda: ["delivered"])
    customer_states:  list[str]  = field(default_factory=list)
    seller_states:    list[str]  = field(default_factory=list)
    categories:       list[str]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sidebar renderer
# ---------------------------------------------------------------------------

def render_sidebar_filters(
    master_df:  pd.DataFrame,
    item_df:    pd.DataFrame,
) -> FilterState:
    """
    Render sidebar filter widgets and return the resulting FilterState.

    Widget keys already present in st.session_state are preserved by
    Streamlit across reruns (the `value=` argument only seeds the FIRST
    render); no defaults are force-overwritten on an existing user
    selection.

    Parameters
    ----------
    master_df : pd.DataFrame
        The full (unfiltered) order-level master DataFrame from load_master().
    item_df : pd.DataFrame
        The full (unfiltered) item-level DataFrame from load_item_master().

    Returns
    -------
    FilterState
        All selected filter values as a plain dataclass — no DataFrames inside.
    """
    st.sidebar.header("Filters")

    # ── 1. Analysis period ───────────────────────────────────────────────
    st.sidebar.subheader("Analysis Period")

    col_start, col_end = st.sidebar.columns(2)
    with col_start:
        date_start = st.date_input(
            "From",
            value=_ANALYSIS_START,
            min_value=_FULL_START,
            max_value=_ANALYSIS_END,
            key="filter_date_start",
        )
    with col_end:
        date_end = st.date_input(
            "To",
            value=_ANALYSIS_END,
            min_value=_FULL_START,
            max_value=_ANALYSIS_END,
            key="filter_date_end",
        )

    # Safe handling of an inverted range: warn, use an effective swapped
    # range for filtering, but do NOT wipe the rest of the user's selections.
    effective_start, effective_end = date_start, date_end
    if date_start > date_end:
        st.sidebar.warning(
            "'From' date is after 'To' date — using the effective range "
            f"{date_end} to {date_start} for filtering."
        )
        effective_start, effective_end = date_end, date_start

    include_2016 = st.sidebar.toggle(
        "Include 2016 partial period",
        value=False,
        key="filter_include_2016",
        help=(
            "Includes the observed partial 2016 records when the selected "
            "date range contains them. The dataset has 329 orders from "
            "Sep-Dec 2016 (4 in Sep, 324 in Oct, 1 in Dec). When off, any "
            "order with a purchase date before 2017-01-01 is excluded, "
            "regardless of the selected 'From' date."
        ),
    )

    # ── 2. Order status ─────────────────────────────────────────────────
    st.sidebar.subheader("Order Status")
    order_statuses = st.sidebar.multiselect(
        "Include statuses",
        options=_ALL_STATUSES,
        default=["delivered"],
        format_func=_format_status_label,
        key="filter_order_statuses",
        help=(
            "Delivery and delay metrics are only meaningful for 'delivered' orders. "
            "Widen the selection for cancellation or funnel analysis."
        ),
    )

    # ── 3. Customer state ───────────────────────────────────────────────
    st.sidebar.subheader("Customer Region")
    available_cstates: list[str] = _get_customer_states(master_df)
    customer_states = st.sidebar.multiselect(
        "Customer state (all if empty)",
        options=available_cstates,
        default=[],
        key="filter_customer_states",
        placeholder="All states",
    )

    # ── 4. Seller state ─────────────────────────────────────────────────
    st.sidebar.subheader("Seller Region")
    available_sstates: list[str] = _get_seller_states(item_df)
    seller_states = st.sidebar.multiselect(
        "Seller state (all if empty)",
        options=available_sstates,
        default=[],
        key="filter_seller_states",
        placeholder="All states",
    )

    # ── 5. Product category ─────────────────────────────────────────────
    st.sidebar.subheader("Product Category")
    available_cats: list[str] = _get_categories(item_df)
    categories = st.sidebar.multiselect(
        "Category (all if empty)",
        options=available_cats,
        default=[],
        format_func=_format_category_label,
        key="filter_categories",
        placeholder="All categories",
    )

    # ── 6. Reset button ─────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.button(
        "Reset all filters",
        key="filter_reset",
        on_click=_reset_filters,
    )

    return FilterState(
        date_start=effective_start,
        date_end=effective_end,
        include_2016=include_2016,
        order_statuses=list(order_statuses),
        customer_states=list(customer_states),
        seller_states=list(seller_states),
        categories=list(categories),
    )


# ---------------------------------------------------------------------------
# Internal shared masking logic
# ---------------------------------------------------------------------------

def _date_and_period_mask(df: pd.DataFrame, fs: FilterState) -> pd.Series:
    """
    Build the shared date-range + 2016-inclusion boolean mask.

    order_purchase_timestamp is the sole authority. No helper flag column
    (is_2016_partial / is_primary_analysis_period) is required or relied
    upon, so the mask behaves correctly even if those columns are absent,
    stale, or mis-populated.
    """
    ts = pd.to_datetime(df["order_purchase_timestamp"])

    range_start = pd.Timestamp(fs.date_start)
    range_end = pd.Timestamp(fs.date_end).replace(hour=23, minute=59, second=59)

    mask = (ts >= range_start) & (ts <= range_end)

    if not fs.include_2016:
        mask &= ts >= _ANALYSIS_START_TS

    return mask


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

def apply_master_filters(
    df: pd.DataFrame,
    fs: FilterState,
) -> pd.DataFrame:
    """
    Apply FilterState to the order-level master DataFrame.

    Filters applied (in order)
    --------------------------
    1. Date range on order_purchase_timestamp, combined with the 2016
       cutoff (order_purchase_timestamp < 2017-01-01 excluded unless
       fs.include_2016 is True) — timestamp-driven, not flag-column-driven.
    2. order_status membership (skipped when fs.order_statuses is empty).
    3. customer_state membership (skipped when fs.customer_states is empty).

    seller_state and categories are NOT applied to the master because those
    fields are item-level. Applying them here would silently drop orders with
    mixed or null category values, violating the two-grain architecture.

    Returns
    -------
    pd.DataFrame
        A filtered copy; the original is never mutated.
    """
    mask = _date_and_period_mask(df, fs)

    if fs.order_statuses:
        mask &= df["order_status"].isin(fs.order_statuses)

    if fs.customer_states:
        mask &= df["customer_state"].isin(fs.customer_states)

    return df.loc[mask].copy()


def apply_item_filters(
    df: pd.DataFrame,
    fs: FilterState,
) -> pd.DataFrame:
    """
    Apply FilterState to the item-level master DataFrame.

    Filters applied (in order)
    --------------------------
    1. Date range on order_purchase_timestamp, combined with the 2016
       cutoff (timestamp-driven, same rule as apply_master_filters).
    2. order_status membership (skipped when fs.order_statuses is empty).
    3. customer_state membership (skipped when fs.customer_states is empty).
    4. seller_state membership (skipped when fs.seller_states is empty).
    5. product_category_name_english membership (skipped when fs.categories is empty).

    Returns
    -------
    pd.DataFrame
        A filtered copy; the original is never mutated.
    """
    mask = _date_and_period_mask(df, fs)

    if fs.order_statuses:
        mask &= df["order_status"].isin(fs.order_statuses)

    if fs.customer_states:
        mask &= df["customer_state"].isin(fs.customer_states)

    if fs.seller_states:
        mask &= df["seller_state"].isin(fs.seller_states)

    if fs.categories:
        mask &= df["product_category_name_english"].isin(fs.categories)

    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Internal helpers — cached option-list builders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _get_customer_states(df: pd.DataFrame) -> list[str]:
    """Sorted list of customer states present in the master."""
    return sorted(s for s in df["customer_state"].dropna().unique().tolist() if s)


@st.cache_data(show_spinner=False)
def _get_seller_states(df: pd.DataFrame) -> list[str]:
    """Sorted list of seller states present in the item master."""
    return sorted(s for s in df["seller_state"].dropna().unique().tolist() if s)


@st.cache_data(show_spinner=False)
def _get_categories(df: pd.DataFrame) -> list[str]:
    """Sorted list of non-null categories present in the item master."""
    return sorted(
        c for c in df["product_category_name_english"].dropna().unique().tolist()
        if c  # exclude empty strings
    )


# ---------------------------------------------------------------------------
# Internal self-tests (synthetic data, no Streamlit runtime required)
# Run directly: python -m src.filters
# ---------------------------------------------------------------------------

def _build_synthetic_master() -> pd.DataFrame:
    rows = []
    # 2016 partial-period rows: 4 Sep, 324 Oct (collapsed to few reps + count),
    # 1 Dec — kept small but date-accurate for boundary testing.
    for d, n in [("2016-09-15", 2), ("2016-10-10", 3), ("2016-12-20", 1)]:
        for i in range(n):
            rows.append({
                "order_id": f"o2016_{d}_{i}",
                "order_purchase_timestamp": pd.Timestamp(d) + pd.Timedelta(hours=i),
                "order_status": "delivered",
                "customer_state": "SP",
            })
    # 2017 rows
    for i in range(5):
        rows.append({
            "order_id": f"o2017jan_{i}",
            "order_purchase_timestamp": pd.Timestamp("2017-01-10") + pd.Timedelta(hours=i),
            "order_status": "delivered" if i % 2 == 0 else "canceled",
            "customer_state": "SP" if i % 2 == 0 else "RJ",
        })
    for i in range(3):
        rows.append({
            "order_id": f"o2018_{i}",
            "order_purchase_timestamp": pd.Timestamp("2018-06-01") + pd.Timedelta(hours=i),
            "order_status": "delivered",
            "customer_state": "MG",
        })
    return pd.DataFrame(rows)


def _build_synthetic_items(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in master.iterrows():
        rows.append({
            "order_id": r["order_id"],
            "order_purchase_timestamp": r["order_purchase_timestamp"],
            "order_status": r["order_status"],
            "customer_state": r["customer_state"],
            "seller_state": "SP",
            "product_category_name_english": "toys",
        })
    return pd.DataFrame(rows)


def _run_self_tests() -> None:
    master = _build_synthetic_master()
    items = _build_synthetic_items(master)

    # TEST 1: defaults
    fs = FilterState()
    assert fs.date_start == _ANALYSIS_START
    assert fs.date_end == _ANALYSIS_END
    assert fs.include_2016 is False
    assert fs.order_statuses == ["delivered"]

    # TEST 2: 2016 OFF, wide range -> no pre-2017 rows
    fs2 = FilterState(date_start=_FULL_START, date_end=_ANALYSIS_END,
                       include_2016=False, order_statuses=[])
    out2 = apply_master_filters(master, fs2)
    assert (out2["order_purchase_timestamp"] >= _ANALYSIS_START_TS).all()

    # TEST 3: 2016 ON, wide range -> 2016 rows present
    fs3 = FilterState(date_start=_FULL_START, date_end=_ANALYSIS_END,
                       include_2016=True, order_statuses=[])
    out3 = apply_master_filters(master, fs3)
    assert (out3["order_purchase_timestamp"] < _ANALYSIS_START_TS).any()

    # TEST 4: ON vs OFF must differ
    assert len(out3) > len(out2)

    # TEST 5: date-only (Jan 2017)
    fs5 = FilterState(date_start=date(2017, 1, 1), date_end=date(2017, 1, 31),
                       include_2016=False, order_statuses=[])
    out5 = apply_master_filters(master, fs5)
    assert out5["order_purchase_timestamp"].dt.month.eq(1).all()
    assert out5["order_purchase_timestamp"].dt.year.eq(2017).all()

    # TEST 6: status filter
    fs6 = FilterState(date_start=_FULL_START, date_end=_ANALYSIS_END,
                       include_2016=True, order_statuses=["canceled"])
    out6 = apply_master_filters(master, fs6)
    assert (out6["order_status"] == "canceled").all()

    # TEST 7: customer state
    fs7 = FilterState(date_start=_FULL_START, date_end=_ANALYSIS_END,
                       include_2016=True, order_statuses=[], customer_states=["RJ"])
    out7 = apply_master_filters(master, fs7)
    assert (out7["customer_state"] == "RJ").all()

    # TEST 8: seller state (item grain)
    fs8 = FilterState(date_start=_FULL_START, date_end=_ANALYSIS_END,
                       include_2016=True, order_statuses=[], seller_states=["SP"])
    out8 = apply_item_filters(items, fs8)
    assert (out8["seller_state"] == "SP").all()

    # TEST 9: category (item grain)
    fs9 = FilterState(date_start=_FULL_START, date_end=_ANALYSIS_END,
                       include_2016=True, order_statuses=[], categories=["toys"])
    out9 = apply_item_filters(items, fs9)
    assert (out9["product_category_name_english"] == "toys").all()

    # TEST 10: combined
    fs10 = FilterState(date_start=_FULL_START, date_end=_ANALYSIS_END,
                        include_2016=True, order_statuses=["delivered"],
                        customer_states=["SP"], seller_states=["SP"],
                        categories=["toys"])
    out10 = apply_item_filters(items, fs10)
    assert (out10["order_status"] == "delivered").all()
    assert (out10["customer_state"] == "SP").all()
    assert (out10["seller_state"] == "SP").all()
    assert (out10["product_category_name_english"] == "toys").all()

    # TEST 11: impossible combination -> empty, no crash
    fs11 = FilterState(date_start=_FULL_START, date_end=_ANALYSIS_END,
                        include_2016=True, order_statuses=["unavailable"],
                        customer_states=["ZZ"])
    out11 = apply_master_filters(master, fs11)
    assert len(out11) == 0

    # TEST 12: original untouched
    master_copy = master.copy(deep=True)
    _ = apply_master_filters(master, fs10)
    _ = apply_item_filters(items, fs10)
    pd.testing.assert_frame_equal(master, master_copy)

    print("All internal self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()


# ### `kpis(1).py` — Reusable KPI calculations

# In[ ]:


"""
kpis.py
=======
Reusable, null-safe KPI computation functions for the Olist analytics platform.

Design rules
------------
- Every function accepts a pre-filtered DataFrame and returns a plain Python
  scalar or a pandas DataFrame/Series.  No Streamlit calls.
- Null values are explicitly excluded; the effective denominator is documented
  for every rate/average so callers can display it in the UI.
- The two-grain architecture is strictly respected:
    * Master-grain functions receive the order-level master DataFrame.
    * Item-grain functions receive the order-item-level DataFrame.
    * Category-level delivery/review functions receive the output of
      build_order_category_table() — one row per order.
- No joins between the two DataFrames are performed here.
- No profit, margin, or COGS calculations are present.
- delay_days sign convention: negative = arrived early.
  "Average delay" always uses only rows where delay_days > 0.

Function groups
---------------
  Sales & Orders     — total_orders, total_sales_value, total_freight_value,
                       avg_order_value, median_order_value, cancellation_rate,
                       monthly_orders_and_sales, orders_by_status
  Customers          — unique_customers, repeat_customer_rate, orders_per_customer
  Delivery           — avg_delivery_days, median_delivery_days, on_time_rate,
                       delay_rate, avg_late_delay_days, delivery_by_state,
                       delivery_by_month
  Category delivery  — delivery_by_category   (input: order-category table)
  Reviews            — avg_review_score, review_coverage, review_score_distribution,
                       low_score_rate, five_star_rate, review_by_state,
                       review_vs_delay
  Category reviews   — review_by_category     (input: order-category table)
 Payments           — payment_type_share, installment_distribution,
                       aov_by_installment_band, payment_reconciliation
 Regional           — sales_by_state, sales_by_seller_state
 Category sales     — sales_by_category, sales_by_seller, items_by_month
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_mean(series: pd.Series) -> float:
    """Mean of non-null values; returns 0.0 if series is empty after dropna."""
    s = series.dropna()
    return float(s.mean()) if len(s) > 0 else 0.0


def _safe_median(series: pd.Series) -> float:
    """Median of non-null values; returns 0.0 if empty."""
    s = series.dropna()
    return float(s.median()) if len(s) > 0 else 0.0


def _safe_sum(series: pd.Series) -> float:
    """Sum of non-null values; returns 0.0 if empty."""
    return float(series.sum(skipna=True))


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, or 0.0 if denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


# ===========================================================================
# Sales & Orders  (master grain)
# ===========================================================================

def total_orders(df: pd.DataFrame) -> int:
    """Count of rows in the filtered master DataFrame (one row = one order)."""
    return len(df)


def total_sales_value(df: pd.DataFrame) -> float:
    """
    Sum of total_price across all orders in df.
    Null values in total_price (775 orders without item data) are excluded.
    """
    return _safe_sum(df["total_price"])


def total_freight_value(df: pd.DataFrame) -> float:
    """Sum of total_freight across all orders in df (null-safe)."""
    return _safe_sum(df["total_freight"])


def avg_order_value(df: pd.DataFrame) -> float:
    """Mean of total_price (null-safe)."""
    return _safe_mean(df["total_price"])


def median_order_value(df: pd.DataFrame) -> float:
    """Median of total_price (null-safe)."""
    return _safe_median(df["total_price"])


def cancellation_rate(df: pd.DataFrame) -> float:
    """
    Proportion of orders with order_status == 'canceled'.

    Intended use: pass the unfiltered (all-status) master for this period
    so the denominator includes all order states.
    Returns 0.0 if df is empty.
    """
    if len(df) == 0:
        return 0.0
    n_canceled = (df["order_status"] == "canceled").sum()
    return _safe_rate(int(n_canceled), len(df))


def monthly_orders_and_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly trend table.  Groups by order_month_str and computes:
      order_count     — number of orders
      total_sales     — SUM(total_price), null-safe
      avg_order_value — MEAN(total_price), null-safe

    Returns rows sorted chronologically by order_month_str.
    Months with zero orders after filtering are not included.
    """
    grouped = (
        df.groupby("order_month_str", sort=True)
        .agg(
            order_count=("order_id", "count"),
            total_sales=("total_price", "sum"),
            avg_order_value=("total_price", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values("order_month_str").reset_index(drop=True)
    return grouped


def orders_by_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order count and percentage share by order_status.
    Returns a DataFrame with columns: order_status, order_count, share_pct.
    Sorted by order_count descending.
    """
    counts = (
        df["order_status"]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "order_count"})
    )
    counts["share_pct"] = counts["order_count"] / counts["order_count"].sum() * 100
    return counts


# ===========================================================================
# Customers  (master grain)
# ===========================================================================

def unique_customers(df: pd.DataFrame) -> int:
    """Count of distinct customer_unique_id values in df."""
    return int(df["customer_unique_id"].nunique())


def repeat_customer_rate(df: pd.DataFrame) -> float:
    """
    Proportion of unique customers who placed more than one order
    within the filtered period.
    Returns 0.0 if no customers.
    """
    if len(df) == 0:
        return 0.0
    orders_per_cust = df.groupby("customer_unique_id")["order_id"].count()
    n_repeat = int((orders_per_cust > 1).sum())
    return _safe_rate(n_repeat, len(orders_per_cust))


def orders_per_customer(df: pd.DataFrame) -> float:
    """
    Average number of orders per unique customer in the filtered period.
    Returns 0.0 if no customers.
    """
    n_cust = df["customer_unique_id"].nunique()
    return _safe_rate(len(df), n_cust)


# ===========================================================================
# Delivery  (master grain — delivered orders only recommended)
# ===========================================================================

def avg_delivery_days(df: pd.DataFrame) -> float:
    """
    Mean of delivery_time_days.
    Denominator: rows where delivery_time_days is not null.
    """
    return _safe_mean(df["delivery_time_days"])


def median_delivery_days(df: pd.DataFrame) -> float:
    """Median of delivery_time_days (null-safe)."""
    return _safe_median(df["delivery_time_days"])


def on_time_rate(df: pd.DataFrame) -> float:
    """
    Proportion of orders that were NOT delayed.
    Denominator: rows where is_delayed is not null (i.e. delivery date known).
    """
    known = df["is_delayed"].dropna()
    if len(known) == 0:
        return 0.0
    return _safe_rate(int((known == 0).sum()), len(known))


def delay_rate(df: pd.DataFrame) -> float:
    """
    Proportion of orders that were delayed (is_delayed == 1).
    Denominator: rows where is_delayed is not null.
    """
    known = df["is_delayed"].dropna()
    if len(known) == 0:
        return 0.0
    return _safe_rate(int((known == 1).sum()), len(known))


def avg_late_delay_days(df: pd.DataFrame) -> float:
    """
    Mean of delay_days for orders where is_delayed == 1.

    Uses is_delayed as the authoritative late flag rather than delay_days > 0
    because 1,292 orders are flagged is_delayed=1 with delay_days <= 0 (edge
    cases where the flag and the computed day difference disagree slightly).
    Using is_delayed==1 as the filter matches the business definition of
    'late order' and aligns with the on_time_rate / delay_rate denominators.

    Returns 0.0 if no late orders are present.
    """
    late = df.loc[df["is_delayed"] == 1, "delay_days"].dropna()
    return _safe_mean(late)


def delivery_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-state delivery summary.  Input: order-level master (delivered orders).

    Returns DataFrame sorted by avg_delivery_days descending, with columns:
      customer_state, order_count, avg_delivery_days, median_delivery_days,
      delay_rate, n_delayed, n_on_time
    """
    # Work only with rows that have a delivery time
    sub = df.dropna(subset=["delivery_time_days"]).copy()

    agg = (
        sub.groupby("customer_state", observed=True)
        .agg(
            order_count=("order_id", "count"),
            avg_delivery_days=("delivery_time_days", "mean"),
            median_delivery_days=("delivery_time_days", "median"),
        )
        .reset_index()
    )

    # Delay rate per state — denominator: orders with known is_delayed
    delay_sub = df.dropna(subset=["is_delayed"]).copy()
    delay_agg = (
        delay_sub.groupby("customer_state", observed=True)
        .apply(
            lambda g: pd.Series({
                "n_delayed": int((g["is_delayed"] == 1).sum()),
                "n_on_time": int((g["is_delayed"] == 0).sum()),
                "n_known":   int(g["is_delayed"].notna().sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    # Guard: groupby on empty input returns no columns
    for col in ["n_delayed", "n_on_time", "n_known"]:
        if col not in delay_agg.columns:
            delay_agg[col] = 0
    delay_agg["delay_rate"] = delay_agg["n_delayed"] / delay_agg["n_known"].replace(0, np.nan)

    result = agg.merge(delay_agg[["customer_state", "n_delayed", "n_on_time", "delay_rate"]],
                       on="customer_state", how="left")
    result = result.sort_values("avg_delivery_days", ascending=False).reset_index(drop=True)
    return result


def delivery_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly delivery trend.  Input: order-level master (delivered orders).

    Returns DataFrame sorted chronologically with columns:
      order_month_str, order_count, avg_delivery_days, delay_rate
    """
    sub = df.dropna(subset=["delivery_time_days"]).copy()
    agg = (
        sub.groupby("order_month_str", sort=True)
        .agg(
            order_count=("order_id", "count"),
            avg_delivery_days=("delivery_time_days", "mean"),
        )
        .reset_index()
    )

    delay_sub = df.dropna(subset=["is_delayed"]).copy()
    if len(delay_sub) > 0:
        # Vectorised: count late flag using a bool column, then divide
        delay_sub = delay_sub.copy()
        delay_sub["_is_late"] = (delay_sub["is_delayed"] == 1).astype(int)
        delay_agg = (
            delay_sub.groupby("order_month_str", sort=True)
            .agg(n_late=("_is_late", "sum"), n_known=("_is_late", "count"))
            .reset_index()
        )
        delay_agg["delay_rate"] = delay_agg["n_late"] / delay_agg["n_known"].replace(0, np.nan)
        delay_agg = delay_agg[["order_month_str", "delay_rate"]]
    else:
        delay_agg = pd.DataFrame(columns=["order_month_str", "delay_rate"])

    result = agg.merge(delay_agg, on="order_month_str", how="left")
    return result.sort_values("order_month_str").reset_index(drop=True)


# ===========================================================================
# Category-level delivery  (order-category table grain)
# ===========================================================================

def delivery_by_category(oct_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-category delivery summary using the order-category attribution table
    produced by build_order_category_table().

    Input grain: one row per order (primary_category already assigned).
    Denominator: unique orders with non-null delivery_time_days.

    Returns DataFrame sorted by delay_rate descending, with columns:
      primary_category, order_count, avg_delivery_days, median_delivery_days,
      delay_rate, n_delayed, n_on_time
    """
    sub = oct_df.dropna(subset=["delivery_time_days"]).copy()

    agg = (
        sub.groupby("primary_category", sort=False)
        .agg(
            order_count=("order_id", "count"),
            avg_delivery_days=("delivery_time_days", "mean"),
            median_delivery_days=("delivery_time_days", "median"),
        )
        .reset_index()
    )

    delay_sub = oct_df.dropna(subset=["is_delayed"]).copy()
    delay_agg = (
        delay_sub.groupby("primary_category", sort=False)
        .apply(
            lambda g: pd.Series({
                "n_delayed": int((g["is_delayed"] == 1).sum()),
                "n_on_time": int((g["is_delayed"] == 0).sum()),
                "n_known":   int(g["is_delayed"].notna().sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    # Guard: groupby on empty input returns no columns
    for col in ["n_delayed", "n_on_time", "n_known"]:
        if col not in delay_agg.columns:
            delay_agg[col] = 0
    delay_agg["delay_rate"] = delay_agg["n_delayed"] / delay_agg["n_known"].replace(0, np.nan)

    result = agg.merge(delay_agg[["primary_category", "n_delayed", "n_on_time", "delay_rate"]],
                       on="primary_category", how="left")
    result = result.sort_values("delay_rate", ascending=False).reset_index(drop=True)
    return result


# ===========================================================================
# Reviews  (master grain)
# ===========================================================================

def avg_review_score(df: pd.DataFrame) -> float:
    """Mean review_score (null-safe)."""
    return _safe_mean(df["review_score"])


def review_coverage(df: pd.DataFrame) -> float:
    """
    Proportion of orders that have a review_score.
    Denominator: all orders in df (including those without reviews).
    """
    if len(df) == 0:
        return 0.0
    n_with_review = int(df["review_score"].notna().sum())
    return _safe_rate(n_with_review, len(df))


def review_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count and percentage of reviews for each score (1–5).
    Returns a DataFrame with columns: review_score, count, share_pct.
    Rows for scores absent from the data are included with count=0.
    """
    known = df["review_score"].dropna()
    counts = known.value_counts().reindex([1.0, 2.0, 3.0, 4.0, 5.0], fill_value=0)
    total = counts.sum()
    result = counts.reset_index()
    result.columns = ["review_score", "count"]
    result["share_pct"] = result["count"] / total * 100 if total > 0 else 0.0
    result["review_score"] = result["review_score"].astype(int)
    return result.sort_values("review_score").reset_index(drop=True)


def low_score_rate(df: pd.DataFrame) -> float:
    """
    Proportion of reviews with score <= 2 (1-star or 2-star).
    Denominator: reviews with non-null review_score.
    """
    known = df["review_score"].dropna()
    if len(known) == 0:
        return 0.0
    return _safe_rate(int((known <= 2).sum()), len(known))


def five_star_rate(df: pd.DataFrame) -> float:
    """
    Proportion of reviews with score == 5.
    Denominator: reviews with non-null review_score.
    """
    known = df["review_score"].dropna()
    if len(known) == 0:
        return 0.0
    return _safe_rate(int((known == 5).sum()), len(known))


def review_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-state review summary.  Input: order-level master.

    Returns DataFrame sorted by avg_review_score ascending (worst first), with:
      customer_state, order_count, avg_review_score, low_score_rate,
      five_star_rate, n_reviews
    """
    sub = df.dropna(subset=["review_score"]).copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["customer_state", "order_count",
                                     "avg_review_score", "n_reviews",
                                     "low_score_rate", "five_star_rate"])
    sub["_low"]  = (sub["review_score"] <= 2).astype(int)
    sub["_five"] = (sub["review_score"] == 5).astype(int)
    result = (
        sub.groupby("customer_state", observed=True)
        .agg(
            order_count=("order_id", "count"),
            avg_review_score=("review_score", "mean"),
            n_reviews=("review_score", "count"),
            _n_low=("_low", "sum"),
            _n_five=("_five", "sum"),
        )
        .reset_index()
    )
    result["low_score_rate"] = result["_n_low"]  / result["n_reviews"].replace(0, np.nan)
    result["five_star_rate"] = result["_n_five"] / result["n_reviews"].replace(0, np.nan)
    return result.drop(columns=["_n_low", "_n_five"]).sort_values(
        "avg_review_score", ascending=True).reset_index(drop=True)


def review_vs_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare review score distribution for on-time vs delayed orders.
    Input: order-level master (delivered orders with both review_score
    and is_delayed non-null).

    Returns DataFrame with columns:
      is_delayed (0/1), order_count, avg_review_score,
      low_score_rate, five_star_rate
    """
    sub = df.dropna(subset=["review_score", "is_delayed"]).copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["is_delayed", "order_count",
                                     "avg_review_score", "low_score_rate", "five_star_rate"])
    sub["_low"]  = (sub["review_score"] <= 2).astype(int)
    sub["_five"] = (sub["review_score"] == 5).astype(int)
    result = (
        sub.groupby("is_delayed", observed=True)
        .agg(
            order_count=("order_id", "count"),
            avg_review_score=("review_score", "mean"),
            _n_low=("_low", "sum"),
            _n_five=("_five", "sum"),
            _n_total=("_low", "count"),
        )
        .reset_index()
    )
    result["low_score_rate"] = result["_n_low"]  / result["_n_total"].replace(0, np.nan)
    result["five_star_rate"] = result["_n_five"] / result["_n_total"].replace(0, np.nan)
    return result.drop(columns=["_n_low", "_n_five", "_n_total"]).sort_values(
        "is_delayed").reset_index(drop=True)


# ===========================================================================
# Category-level reviews  (order-category table grain)
# ===========================================================================

def review_by_category(oct_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-category review summary using the order-category attribution table.

    Input grain: one row per order (primary_category already assigned).
    Denominator: orders with non-null review_score.

    Returns DataFrame sorted by avg_review_score ascending (worst first):
      primary_category, order_count, avg_review_score,
      low_score_rate, five_star_rate, n_reviews
    """
    sub = oct_df.dropna(subset=["review_score"]).copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["primary_category", "order_count",
                                     "avg_review_score", "n_reviews",
                                     "low_score_rate", "five_star_rate"])
    sub["_low"]  = (sub["review_score"] <= 2).astype(int)
    sub["_five"] = (sub["review_score"] == 5).astype(int)
    result = (
        sub.groupby("primary_category", sort=False)
        .agg(
            order_count=("order_id", "count"),
            avg_review_score=("review_score", "mean"),
            n_reviews=("review_score", "count"),
            _n_low=("_low", "sum"),
            _n_five=("_five", "sum"),
        )
        .reset_index()
    )
    result["low_score_rate"] = result["_n_low"]  / result["n_reviews"].replace(0, np.nan)
    result["five_star_rate"] = result["_n_five"] / result["n_reviews"].replace(0, np.nan)
    return result.drop(columns=["_n_low", "_n_five"]).sort_values(
        "avg_review_score", ascending=True).reset_index(drop=True)


# ===========================================================================
# Payments  (master grain)
# ===========================================================================

def payment_type_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-payment-type summary.  Input: order-level master.

    Returns DataFrame sorted by order_count descending:
      primary_payment_type, order_count, share_pct, avg_order_value,
      median_order_value
    """
    sub = df.dropna(subset=["primary_payment_type"]).copy()
    agg = (
        sub.groupby("primary_payment_type", observed=True)
        .agg(
            order_count=("order_id", "count"),
            avg_order_value=("total_price", "mean"),
            median_order_value=("total_price", "median"),
        )
        .reset_index()
    )
    agg["avg_order_value"]    = agg["avg_order_value"].fillna(0.0)
    agg["median_order_value"] = agg["median_order_value"].fillna(0.0)
    agg["share_pct"] = agg["order_count"] / agg["order_count"].sum() * 100
    return agg.sort_values("order_count", ascending=False).reset_index(drop=True)


def installment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution of max_installments for credit-card orders only.

    Returns DataFrame with columns: max_installments (int), order_count,
    share_pct.  Installments > 12 are grouped as '13+'.
    """
    cc = df[df["primary_payment_type"] == "credit_card"].copy()
    cc = cc.dropna(subset=["max_installments"])
    cc["installment_band"] = cc["max_installments"].clip(lower=1, upper=12).astype(int)
    counts = cc["installment_band"].value_counts().sort_index().reset_index()
    counts.columns = ["max_installments", "order_count"]
    total = counts["order_count"].sum()
    counts["share_pct"] = counts["order_count"] / total * 100 if total > 0 else 0.0
    return counts


def aov_by_installment_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average order value (total_price) by installment band for credit-card
    orders.

    Bands: 1 / 2–3 / 4–6 / 7–12 / 13+
    Returns DataFrame with columns: band_label, order_count, avg_order_value.
    """
    cc = df[df["primary_payment_type"] == "credit_card"].copy()
    cc = cc.dropna(subset=["max_installments", "total_price"])

    def _band(x: float) -> str:
        if x <= 1:   return "1"
        if x <= 3:   return "2–3"
        if x <= 6:   return "4–6"
        if x <= 12:  return "7–12"
        return "13+"

    cc["band_label"] = cc["max_installments"].apply(_band)
    band_order = ["1", "2–3", "4–6", "7–12", "13+"]
    agg = (
        cc.groupby("band_label", sort=False)
        .agg(order_count=("order_id", "count"),
             avg_order_value=("total_price", "mean"))
        .reset_index()
    )
    agg["band_label"] = pd.Categorical(agg["band_label"], categories=band_order, ordered=True)
    return agg.sort_values("band_label").reset_index(drop=True)


def payment_reconciliation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Payment-vs-sales reconciliation by payment type.

    Shows the average difference between total_payment_value and total_price
    per payment type.  Labelled as a reconciliation metric only — no causal
    attribution is made.

    Returns DataFrame with columns:
      primary_payment_type, order_count, avg_total_price,
      avg_total_payment_value, avg_difference
    """
    sub = df.dropna(subset=["primary_payment_type", "total_payment_value", "total_price"]).copy()
    agg = (
        sub.groupby("primary_payment_type", observed=True)
        .agg(
            order_count=("order_id", "count"),
            avg_total_price=("total_price", "mean"),
            avg_total_payment_value=("total_payment_value", "mean"),
        )
        .reset_index()
    )
    agg["avg_difference"] = agg["avg_total_payment_value"] - agg["avg_total_price"]
    return agg.sort_values("order_count", ascending=False).reset_index(drop=True)


# ===========================================================================
# Regional  (master grain for customer state; item grain for seller state)
# ===========================================================================

def sales_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer-state sales summary.  Input: order-level master.

    Returns DataFrame sorted by total_sales descending:
      customer_state, order_count, total_sales, avg_order_value,
      total_freight, unique_customers
    """
    sub = df.dropna(subset=["customer_state"]).copy()
    agg = (
        sub.groupby("customer_state", observed=True)
        .agg(
            order_count=("order_id", "count"),
            total_sales=("total_price", "sum"),
            avg_order_value=("total_price", "mean"),
            total_freight=("total_freight", "sum"),
            unique_customers=("customer_unique_id", "nunique"),
        )
        .reset_index()
    )
    return agg.sort_values("total_sales", ascending=False).reset_index(drop=True)


def sales_by_seller_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-seller-state sales summary.  Input: item-level master.

    Uses item-level price for revenue attribution.

    Returns DataFrame sorted by total_sales_value descending:
      seller_state, item_count, unique_sellers, unique_orders,
      total_sales_value, avg_item_price, avg_freight_per_item
    """
    sub = df.dropna(subset=["seller_state"]).copy()
    agg = (
        sub.groupby("seller_state", observed=True)
        .agg(
            item_count=("order_item_id", "count"),
            unique_sellers=("seller_id", "nunique"),
            unique_orders=("order_id", "nunique"),
            total_sales_value=("price", "sum"),
            avg_item_price=("price", "mean"),
            avg_freight_per_item=("freight_value", "mean"),
        )
        .reset_index()
    )
    return agg.sort_values("total_sales_value", ascending=False).reset_index(drop=True)


# ===========================================================================
# Category sales  (item-level grain)
# ===========================================================================

def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-category sales summary.  Input: item-level master.

    Uses item-level price — NOT total_price from the order master —
    to correctly attribute each item's revenue to its category.

    Returns DataFrame sorted by total_sales_value descending:
      product_category_name_english, item_count, unique_orders,
      unique_sellers, total_sales_value, avg_item_price,
      median_item_price, avg_freight_per_item, total_freight_value
    """
    cat_col = "product_category_name_english"
    sub = df.dropna(subset=[cat_col]).copy()
    agg = (
        sub.groupby(cat_col, observed=True)
        .agg(
            item_count=("order_item_id", "count"),
            unique_orders=("order_id", "nunique"),
            unique_sellers=("seller_id", "nunique"),
            total_sales_value=("price", "sum"),
            avg_item_price=("price", "mean"),
            median_item_price=("price", "median"),
            avg_freight_per_item=("freight_value", "mean"),
            total_freight_value=("freight_value", "sum"),
        )
        .reset_index()
    )
    return agg.sort_values("total_sales_value", ascending=False).reset_index(drop=True)


def sales_by_seller(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-seller sales and delivery summary.  Input: item-level master.

    Delivery rate denominator: unique orders per seller with non-null
    is_delayed.  A multi-item order is counted only once per seller.

    Returns DataFrame sorted by total_sales_value descending:
      seller_id, seller_state, item_count, unique_orders,
      total_sales_value, avg_item_price, avg_freight_per_item,
      delay_rate, avg_delivery_days
    """
    agg = (
        df.groupby("seller_id", sort=False)
        .agg(
            seller_state=("seller_state", "first"),
            item_count=("order_item_id", "count"),
            unique_orders=("order_id", "nunique"),
            total_sales_value=("price", "sum"),
            avg_item_price=("price", "mean"),
            avg_freight_per_item=("freight_value", "mean"),
        )
        .reset_index()
    )

    # Delivery metrics: deduplicate to one row per (seller_id, order_id)
    order_level = (
        df.dropna(subset=["is_delayed"])
        .drop_duplicates(subset=["seller_id", "order_id"])
        [["seller_id", "order_id", "is_delayed", "delivery_time_days"]]
    )
    delivery_agg = (
        order_level.groupby("seller_id", sort=False)
        .apply(
            lambda g: pd.Series({
                "delay_rate": _safe_rate(
                    int((g["is_delayed"] == 1).sum()),
                    len(g)
                ),
                "avg_delivery_days": _safe_mean(g["delivery_time_days"]),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    result = agg.merge(delivery_agg, on="seller_id", how="left")
    # Sellers with no delivered orders with resolved is_delayed get NaN from the
    # left merge — fill with 0.0 so delay_rate is always a valid float in [0, 1].
    result["delay_rate"] = result["delay_rate"].fillna(0.0)
    result["avg_delivery_days"] = result["avg_delivery_days"].fillna(0.0)
    return result.sort_values("total_sales_value", ascending=False).reset_index(drop=True)


def items_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly item-level sales trend.  Input: item-level master.

    Returns DataFrame sorted chronologically:
      order_month_str, item_count, total_item_sales_value, avg_item_price
    """
    agg = (
        df.groupby("order_month_str", sort=True)
        .agg(
            item_count=("order_item_id", "count"),
            total_item_sales_value=("price", "sum"),
            avg_item_price=("price", "mean"),
        )
        .reset_index()
    )
    return agg.sort_values("order_month_str").reset_index(drop=True)


# ### `charts(2).py` — Reusable Plotly chart factories

# In[ ]:


"""
charts.py
=========
Reusable Plotly chart factory functions for the Olist analytics platform.

Rules
-----
- Every function returns a plotly.graph_objects.Figure.
- No Streamlit calls inside this module.
- All charts share a consistent visual config via _LAYOUT_DEFAULTS.
- Delay-sign annotation helper is available for any chart showing delay_days.
- top_n truncation is applied before plotting, not after.
- Presentation-only label formatting never changes the underlying data.
- Long categorical labels are wrapped only at the display layer.
- Payment methods use business-friendly display names.
- Dense donut charts use the legend for category names and percentages for slices.
- Internal Plotly titles act as small subtitles; the Streamlit section heading
  is the primary title. Axis titles + automargin handle label overflow instead
  of hardcoded margins.
"""

from __future__ import annotations

import re
import textwrap

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ===========================================================================
# Shared visual configuration
# ===========================================================================

_PALETTE = [
    "#2563EB",  # blue      - sales / revenue
    "#16A34A",  # green     - positive / payments
    "#F59E0B",  # amber     - warning / freight
    "#DC2626",  # red       - delay / risk
    "#7C3AED",  # purple    - customer experience
    "#0891B2",  # cyan      - regional
    "#4F46E5",  # indigo    - seller performance
    "#EA580C",  # orange    - delivery / operations
    "#DB2777",  # pink      - accent
    "#475569",  # slate     - neutral
    "#65A30D",  # lime      - accent
]

# Semantic colors — one metric family = one consistent, colour-blind-friendly hue.
_COLOR_SALES = "#2563EB"        # blue
_COLOR_ORDERS = "#16A34A"       # cyan / teal
_COLOR_ON_TIME = "#16A34A"      # green
_COLOR_DELAY = "#DC2626"        # red
_COLOR_WARNING = "#F59E0B"      # amber
_COLOR_NEUTRAL = "#64748B"      # slate
_COLOR_CUSTOMER = "#7C3AED"     # purple
_COLOR_PAYMENTS = "#16A34A"     # green
_COLOR_REGIONAL = "#0891B2"     # cyan
_COLOR_SELLER = "#4F46E5"       # indigo
_COLOR_DELIVERY = "#EA580C"     # orange

_COLOR_PAYMENT = {
    "credit_card": "#2563EB",
    "boleto": "#DC2626",
    "voucher": "#F59E0B",
    "debit_card": "#16A34A",
    "not_defined": "#64748B",
}

_PAYMENT_LABELS = {
    "credit_card": "Credit Card",
    "boleto": "Boleto",
    "voucher": "Voucher",
    "debit_card": "Debit Card",
    "not_defined": "Not Defined",
}

_TEXT_DARK = "#111827"
_TEXT_MEDIUM = "#4B5563"
_TEXT_MUTED = "#6B7280"
_GRID = "#E5E7EB"

_FONT_FAMILY = "Segoe UI, Arial, sans-serif"


_LAYOUT_DEFAULTS: dict = dict(
    template="plotly_white",

    font=dict(family=_FONT_FAMILY, size=13, color=_TEXT_DARK),

    title=dict(font=dict(family=_FONT_FAMILY, size=13, color=_TEXT_DARK)),

    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",

    margin=dict(l=24, r=24, t=48, b=48),

    xaxis=dict(
        tickfont=dict(family=_FONT_FAMILY, size=11, color=_TEXT_MEDIUM),
        title_font=dict(family=_FONT_FAMILY, size=13, color="#0F172A"),
        tickcolor="#9CA3AF",
        linecolor="#D1D5DB",
        zerolinecolor="#D1D5DB",
        automargin=True,
    ),

    yaxis=dict(
        tickfont=dict(family=_FONT_FAMILY, size=11, color=_TEXT_MEDIUM),
        title_font=dict(family=_FONT_FAMILY, size=13, color="#0F172A"),
        tickcolor="#9CA3AF",
        linecolor="#D1D5DB",
        zerolinecolor="#D1D5DB",
        automargin=True,
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(family=_FONT_FAMILY, size=11, color=_TEXT_DARK),
    ),

    hoverlabel=dict(
        bgcolor="#111827",
        font=dict(family=_FONT_FAMILY, size=12, color="#FFFFFF"),
    ),

    colorway=_PALETTE,
    autosize=True,
)


_DELAY_NOTE = (
    "Note: negative delay_days = order arrived early. "
    "Avg Delay shows late orders (is_delayed = 1) only."
)

_MULTICATEGORY_NOTE = (
    "Multi-category orders attributed to the most frequently purchased category."
)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _base_layout(**overrides) -> dict:
    """Build a layout dict from shared defaults, deep-merging nested dicts."""
    layout = dict(_LAYOUT_DEFAULTS)

    for key, value in overrides.items():
        if key in layout and isinstance(layout[key], dict) and isinstance(value, dict):
            merged = dict(layout[key])
            merged.update(value)

            # Plotly applies axis title fonts reliably when the font is nested
            # under the axis title object. Convert the shorthand title_font
            # setting here so all chart functions use the same visible styling.
            if key.startswith("xaxis") or key.startswith("yaxis"):
                axis_title_font = merged.pop("title_font", None)
                if axis_title_font is not None:
                    axis_title = merged.get("title", "")
                    if isinstance(axis_title, dict):
                        axis_title = dict(axis_title)
                        axis_title["font"] = axis_title_font
                    else:
                        axis_title = dict(text=axis_title, font=axis_title_font)
                    merged["title"] = axis_title

            layout[key] = merged
        else:
            layout[key] = value

    return layout


def _subtitle_title(text: str, size: int = 12) -> dict:
    """
    Internal Plotly titles act as small subtitles (the Streamlit section
    heading is the real title), so they're deliberately smaller/muted
    while still meeting contrast requirements.
    """
    return dict(
        text=text,
        font=dict(family=_FONT_FAMILY, size=size, color=_TEXT_MEDIUM),
        x=0,
        xanchor="left",
    )


def _clean_label(value) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()
    if not text:
        return "Unknown"

    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _pretty_payment_label(value) -> str:
    if pd.isna(value):
        return "Unknown"

    key = re.sub(r"[-\s]+", "_", str(value).strip().lower())

    if key in _PAYMENT_LABELS:
        return _PAYMENT_LABELS[key]

    return _clean_label(value).title()


def _wrap_axis_label(value: str, width: int = 26) -> str:
    text = str(value)
    if len(text) <= width:
        return text

    parts = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return "<br>".join(parts[:3])


def _pretty_category_label(value) -> str:
    """Presentation-only formatter for Olist product categories."""
    text = _clean_label(value)

    replacements = {
        "bed bath table": "Bed & Bath Table",
        "health beauty": "Health & Beauty",
        "fashion bags accessories": "Fashion Bags & Accessories",
        "home appliances": "Home Appliances",
        "home comfort": "Home Comfort",
        "home construction": "Home Construction",
        "home furniture": "Home Furniture",
        "home and kitchen": "Home & Kitchen",
        "computers accessories": "Computers & Accessories",
        "audio": "Audio",
        "baby": "Baby",
        "books general interest": "Books — General Interest",
        "books imported": "Books — Imported",
        "books technical": "Books — Technical",
        "christmas products": "Christmas Products",
        "construction tools construction": "Construction Tools",
        "construction tools garden": "Construction Tools — Garden",
        "construction tools lights": "Construction Tools — Lighting",
        "construction tools safety": "Construction Tools — Safety",
        "cool stuff": "Cool Stuff",
        "diapers and hygiene": "Diapers & Hygiene",
        "dvds blu ray": "DVDs & Blu-ray",
        "electronics": "Electronics",
        "fashion childrens clothes": "Children's Fashion",
        "fashion shoes": "Fashion Shoes",
        "fashion male clothing": "Men's Fashion",
        "fashion female clothing": "Women's Fashion",
        "flowers": "Flowers",
        "food": "Food",
        "food drink": "Food & Drink",
        "furniture bedroom": "Bedroom Furniture",
        "furniture decor": "Furniture Decor",
        "furniture living room": "Living Room Furniture",
        "garden tools": "Garden Tools",
        "industry commerce and business": "Industry, Commerce & Business",
        "kitchen dining laundry garden": "Kitchen, Dining & Garden",
        "luggage accessories": "Luggage & Accessories",
        "market place": "Marketplace",
        "music": "Music",
        "musical instruments": "Musical Instruments",
        "office furniture": "Office Furniture",
        "party supplies": "Party Supplies",
        "perfumery": "Perfumery",
        "pet shop": "Pet Shop",
        "signaling and security": "Signaling & Security",
        "small appliances": "Small Appliances",
        "small appliances home oven and coffee": "Small Appliances (Home & Coffee)",
        "sports leisure": "Sports & Leisure",
        "stationery": "Stationery",
        "tablets printing image": "Tablets, Printing & Image",
        "telephony": "Telephony",
        "toys": "Toys",
        "watches gifts": "Watches & Gifts",
    }

    key = text.lower()
    if key in replacements:
        return replacements[key]

    return text.title()


def _is_uuid_like(value) -> bool:
    if pd.isna(value):
        return False

    text = str(value).strip()
    return bool(re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        r"|[0-9a-fA-F]{32}",
        text,
    ))


def _seller_display_labels(values) -> list[str]:
    """Ranking labels ('Seller 01', 'Seller 02', ...) for hashed seller IDs."""
    return [f"Seller {i:02d}" for i in range(1, len(values) + 1)]


def _format_axis_label(value, column_name: str) -> str:
    column = str(column_name).lower()

    if "seller_id" in column and _is_uuid_like(value):
        return str(value)
    if "product_category" in column or "category" in column:
        return _pretty_category_label(value)
    if "payment" in column or "payment_type" in column:
        return _pretty_payment_label(value)

    return _clean_label(value)


def _format_bar_labels(plot_df: pd.DataFrame, y: str) -> tuple[pd.DataFrame, bool]:
    """
    Prepare display labels for horizontal bar charts.
    Never mutates source data — real seller_id is preserved for hover via customdata.
    """
    result = plot_df.copy()
    column = str(y).lower()
    seller_axis = "seller_id" in column

    if seller_axis:
        original_col = f"__original_{y}"
        result[original_col] = result[y].astype(str)
        result[y] = _seller_display_labels(result[y].tolist())
        return result, True

    result[y] = [_wrap_axis_label(_format_axis_label(v, y), width=28) for v in result[y]]
    return result, False


def _semantic_metric_color(text: str, fallback_index: int = 0) -> str:
    """Pick a semantic color from a chart/series label. Presentation-only."""
    label = str(text).lower()

    if any(w in label for w in ["delay", "late", "cancel", "low score", "1–2 star", "1-2 star", "risk"]):
        return _COLOR_DELAY
    if any(w in label for w in ["on-time", "on time", "5-star", "5 star", "delivered"]):
        return _COLOR_ON_TIME
    if any(w in label for w in ["seller"]):
        return _COLOR_SELLER
    if any(w in label for w in ["region", "state"]):
        return _COLOR_REGIONAL
    if any(w in label for w in ["review", "customer experience", "satisfaction"]):
        return _COLOR_CUSTOMER
    if any(w in label for w in ["payment", "installment"]):
        return _COLOR_PAYMENTS
    if any(w in label for w in ["sales", "revenue", "price", "value", "aov"]):
        return _COLOR_SALES
    if any(w in label for w in ["order", "orders", "items", "activity", "volume"]):
        return _COLOR_ORDERS
    if any(w in label for w in ["freight", "shipping"]):
        return _COLOR_WARNING
    if any(w in label for w in ["delivery", "operations", "carrier"]):
        return _COLOR_DELIVERY

    return _PALETTE[fallback_index % len(_PALETTE)]


def _add_footer_annotation(fig: go.Figure, text: str) -> go.Figure:
    """
    Shared footer-note helper. Positioned below the x-axis while
    remaining visible inside the chart container.
    """
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.20,
        showarrow=False,
        font=dict(family=_FONT_FAMILY, size=10, color=_TEXT_MUTED),
        align="left",
        xanchor="left",
        yanchor="top",
    )

    current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    bottom = max(int(current_margin.get("b", 48)), 90)

    fig.update_layout(
        margin=dict(
            l=current_margin.get("l", 24),
            r=current_margin.get("r", 24),
            t=current_margin.get("t", 48),
            b=bottom,
        )
    )

    return fig


def _add_delay_annotation(fig: go.Figure) -> go.Figure:
    return _add_footer_annotation(fig, _DELAY_NOTE)


def _add_multicategory_annotation(fig: go.Figure) -> go.Figure:
    return _add_footer_annotation(fig, _MULTICATEGORY_NOTE)


# ===========================================================================
# 1. line_trend
# ===========================================================================

def line_trend(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    title: str,
    y_labels: dict[str, str] | None = None,
    secondary_y_col: str | None = None,
    y_axis_title: str = "",
    y2_axis_title: str = "",
    add_delay_note: bool = False,
) -> go.Figure:
    """Multi-series line chart with optional dual Y-axis."""
    y_labels = y_labels or {}
    fig = go.Figure()

    for i, col in enumerate(y_cols):
        on_secondary = col == secondary_y_col
        display_name = y_labels.get(col, _clean_label(col))

        if "avg delivery" in display_name.lower() or "delivery days" in display_name.lower():
            color = _COLOR_SALES
        else:
            color = _semantic_metric_color(display_name, fallback_index=i)

        fig.add_trace(
            go.Scatter(
                x=[
                    _wrap_axis_label(_pretty_category_label(v), width=24)
                    if ("category" in str(x).lower() or "product_category" in str(x).lower())
                    else _pretty_payment_label(v) if "payment" in str(x).lower()
                    else _clean_label(v)
                    for v in df[x]
                ],
                y=df[col],
                name=display_name,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                yaxis="y2" if on_secondary else "y",
                hovertemplate=f"{display_name}: %{{y:,.2f}}<extra></extra>",
            )
        )

    layout_kwargs: dict = dict(
        title=_subtitle_title(title, size=13),
        xaxis=dict(
            title="",
            tickangle=-30,
            tickfont=dict(size=11, color=_TEXT_MEDIUM),
            showgrid=False,
            automargin=True,
        ),
        yaxis=dict(
            title=y_axis_title,
            title_font=dict(size=13, color="#0F172A"),
            title_standoff=8,
            tickfont=dict(size=11, color=_TEXT_MEDIUM),
            showgrid=True,
            gridcolor=_GRID,
            zeroline=False,
            automargin=True,
        ),
    )

    if pd.api.types.is_datetime64_any_dtype(df[x]):
        layout_kwargs["xaxis"]["tickformat"] = "%b %Y"
        layout_kwargs["xaxis"]["tickangle"] = -25

    if secondary_y_col:
        layout_kwargs["yaxis2"] = dict(
            title=y2_axis_title,
            title_font=dict(size=13, color="#0F172A"),
            title_standoff=8,
            tickfont=dict(size=11, color=_TEXT_MEDIUM),
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        )

    fig.update_layout(**_base_layout(**layout_kwargs))

    if add_delay_note:
        fig = _add_delay_annotation(fig)

    return fig


# ===========================================================================
# 2. bar_horizontal
# ===========================================================================

def bar_horizontal(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    top_n: int = 20,
    color: str | None = None,
    x_axis_title: str = "",
    color_discrete_map: dict | None = None,
    add_multicategory_note: bool = False,
) -> go.Figure:
    """
    Horizontal bar chart, sorted descending, clipped to top_n rows.
    Left margin relies on automargin instead of a hardcoded value, so it
    never pushes the chart further right than the labels require.
    """
    plot_df = df.sort_values(x, ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values(x, ascending=True)

    plot_df, seller_axis = _format_bar_labels(plot_df, y)

    semantic_color = _semantic_metric_color(f"{title} {y} {x}", fallback_index=0)
    bar_colors = None if color is not None else [semantic_color] * len(plot_df)

    labels = {x: x_axis_title or _clean_label(x), y: ""}

    fig = px.bar(
        plot_df,
        x=x,
        y=y,
        orientation="h",
        title=title,
        color=color,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=_PALETTE,
        labels=labels,
    )

    if color is None and len(fig.data) > 0:
        fig.update_traces(marker_color=semantic_color)

    if seller_axis:
        original_col = f"__original_{y}"
        if len(fig.data) > 0:
            customdata = plot_df[original_col].to_numpy()
            fig.update_traces(
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Seller ID: %{customdata}<br>"
                    f"{x_axis_title or _clean_label(x)}: %{{x:,.2f}}"
                    "<extra></extra>"
                ),
            )
    else:
        if len(fig.data) > 0:
            fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{x_axis_title or _clean_label(x)}: %{{x:,.2f}}"
                    "<extra></extra>"
                )
            )

    fig.update_layout(
        **_base_layout(
            title=_subtitle_title(title, size=12),
            xaxis=dict(
                title=x_axis_title or _clean_label(x),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=True,
                gridcolor=_GRID,
                zeroline=False,
                automargin=True,
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=11, color=_TEXT_DARK),
                showgrid=False,
                automargin=True,
            ),
            margin=dict(l=24, r=24, t=48, b=48),
            showlegend=color is not None,
        )
    )

    if add_multicategory_note:
        fig = _add_multicategory_annotation(fig)

    return fig


# ===========================================================================
# 3. bar_vertical
# ===========================================================================

def bar_vertical(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    y_axis_title: str = "",
    x_axis_title: str = "",
    text_col: str | None = None,
    color_discrete_sequence: list | None = None,
) -> go.Figure:
    """Professional vertical bar chart."""
    plot_df = df.copy()
    x_lower = str(x).lower()

    if "category" in x_lower or "product_category" in x_lower:
        plot_df[x] = [_wrap_axis_label(_pretty_category_label(v), width=24) for v in plot_df[x]]
    elif "payment" in x_lower:
        plot_df[x] = [_pretty_payment_label(v) for v in plot_df[x]]

    sequence = color_discrete_sequence or _PALETTE

    fig = px.bar(
        plot_df,
        x=x,
        y=y,
        title=title,
        color=color,
        text=text_col,
        color_discrete_sequence=sequence,
        labels={x: x_axis_title or _clean_label(x), y: y_axis_title or _clean_label(y)},
    )

    if color is None and len(fig.data) > 0:
        if "payment" in x_lower:
            payment_colors = [
                _COLOR_PAYMENT.get(
                    re.sub(r"[-\s]+", "_", str(value).strip().lower()),
                    _COLOR_NEUTRAL,
                )
                for value in plot_df[x]
            ]
            fig.update_traces(marker_color=payment_colors)
        elif "credit card" in title.lower():
            fig.update_traces(marker_color=_COLOR_PAYMENT["credit_card"])
        else:
            fig.update_traces(
                marker_color=_semantic_metric_color(
                    f"{title} {y}",
                    fallback_index=0,
                )
            )

    fig.update_traces(textposition="outside", cliponaxis=False)

    fig.update_layout(
        **_base_layout(
            title=_subtitle_title(title, size=12),
            xaxis=dict(
                title=x_axis_title or _clean_label(x),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                tickangle=-30,
                showgrid=False,
                automargin=True,
            ),
            yaxis=dict(
                title=y_axis_title or _clean_label(y),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=True,
                gridcolor=_GRID,
                zeroline=False,
                automargin=True,
            ),
        )
    )

    return fig


# ===========================================================================
# 4. donut_chart
# ===========================================================================

def donut_chart(
    labels: list,
    values: list,
    title: str,
    hole: float = 0.45,
    pull_first: float = 0.0,
) -> go.Figure:
    """
    Professional donut chart.
    - <=5 slices: percent+label inside, readable horizontal legend below.
    - >5 slices: legend (right side) carries names, slices show percent only,
      avoiding label collisions.
    - Payment methods get business-friendly names + semantic colors.
    """
    raw_labels = list(labels)
    display_labels = []

    for value in raw_labels:
        key = str(value).strip().lower()

        if key in _PAYMENT_LABELS:
            display_labels.append(_PAYMENT_LABELS[key])
        else:
            display_labels.append(_clean_label(value).title())

    pull = [pull_first] + [0.0] * (len(raw_labels) - 1) if pull_first else None

    colors = []

    for label in raw_labels:
        key = str(label).strip().lower()

        if key in _COLOR_PAYMENT:
            colors.append(_COLOR_PAYMENT[key])
        elif key in {"on time", "on-time", "ontime"}:
            colors.append(_COLOR_ON_TIME)
        elif key == "late":
            colors.append(_COLOR_DELAY)
        else:
            colors.append(_PALETTE[len(colors) % len(_PALETTE)])

    n_slices = len(display_labels)
    dense = n_slices > 5

    if not dense:
        textinfo = "percent+label"
        textposition = "outside"

        legend = dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(
                family=_FONT_FAMILY,
                size=11,
                color=_TEXT_DARK,
            ),
            bgcolor="rgba(255,255,255,0)",
            traceorder="normal",
        )

        margin = dict(l=24, r=24, t=48, b=90)
        showlegend = True

    else:
        textinfo = "percent"
        textposition = "inside"

        legend = dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(
                family=_FONT_FAMILY,
                size=11,
                color=_TEXT_DARK,
            ),
            bgcolor="rgba(255,255,255,0)",
            traceorder="normal",
        )

        margin = dict(l=24, r=170, t=48, b=24)
        showlegend = True

    fig = go.Figure(
        go.Pie(
            labels=display_labels,
            values=values,
            hole=hole,
            pull=pull,
            marker=dict(
                colors=colors,
                line=dict(
                    color="#FFFFFF",
                    width=1,
                ),
            ),
            textinfo=textinfo,
            textposition=textposition,
            textfont=dict(
                family=_FONT_FAMILY,
                size=11,
                color=_TEXT_DARK if textposition == "outside" else "#FFFFFF",
            ),
            insidetextorientation="horizontal",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Orders: %{value:,}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **_base_layout(
            title=_subtitle_title(title, size=13),
            legend=legend,
            margin=margin,
            showlegend=showlegend,
        )
    )

    return fig


# ===========================================================================
# 5. histogram
# ===========================================================================

def histogram(
    series: pd.Series,
    title: str,
    xaxis_title: str = "",
    nbins: int = 40,
    clip_upper: float | None = None,
    add_delay_note: bool = False,
) -> go.Figure:
    """Histogram for a numeric Series."""
    data = series.dropna().copy()
    clipped = False

    if clip_upper is not None:
        if len(data) > 0 and data.max() >= clip_upper:
            clipped = True
        data = data.clip(upper=clip_upper)

    metric_color = _semantic_metric_color(title, fallback_index=0)

    fig = go.Figure(
        go.Histogram(
            x=data,
            nbinsx=nbins,
            marker_color=metric_color,
            opacity=0.90,
            hovertemplate=(
                f"{xaxis_title or _clean_label(title)}: %{{x}}<br>Orders: %{{y:,}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **_base_layout(
            title=_subtitle_title(title, size=12),
            xaxis=dict(
                title=xaxis_title or _clean_label(title),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=False,
                automargin=True,
            ),
            yaxis=dict(
                title="Orders",
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=True,
                gridcolor=_GRID,
                zeroline=False,
                automargin=True,
            ),
            bargap=0.05,
        )
    )

    if clipped:
        fig.add_annotation(
            text=f"Values above {clip_upper:.0f} clipped for readability",
            xref="paper",
            yref="paper",
            x=1,
            y=1.06,
            showarrow=False,
            font=dict(family=_FONT_FAMILY, size=10, color=_TEXT_MUTED),
            align="right",
        )

    if add_delay_note:
        fig = _add_delay_annotation(fig)

    return fig


# ===========================================================================
# 6. box_plot
# ===========================================================================

def box_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    y_axis_title: str = "",
    x_axis_title: str = "",
    add_delay_note: bool = False,
    max_categories: int = 30,
) -> go.Figure:
    """Box plot with one box per x-category."""
    medians = df.groupby(x, observed=True)[y].median().sort_values(ascending=False)
    top_cats = medians.head(max_categories).index.tolist()

    plot_df = df[df[x].isin(top_cats)].copy()
    plot_df[x] = pd.Categorical(plot_df[x], categories=top_cats, ordered=True)

    if "category" in str(x).lower() or "product_category" in str(x).lower():
        plot_df[x] = plot_df[x].map(_pretty_category_label)

    fig = px.box(
        plot_df.sort_values(x),
        x=x,
        y=y,
        title=title,
        labels={x: x_axis_title or _clean_label(x), y: y_axis_title or _clean_label(y)},
        color_discrete_sequence=_PALETTE,
    )

    fig.update_layout(
        **_base_layout(
            title=_subtitle_title(title, size=12),
            xaxis=dict(
                title=x_axis_title or _clean_label(x),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=10, color=_TEXT_MEDIUM),
                tickangle=-30,
                automargin=True,
            ),
            yaxis=dict(
                title=y_axis_title or _clean_label(y),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=True,
                gridcolor=_GRID,
                zeroline=False,
                automargin=True,
            ),
        )
    )

    if add_delay_note:
        fig = _add_delay_annotation(fig)

    return fig


# ===========================================================================
# 7. scatter_plot
# ===========================================================================

def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    size: str | None = None,
    hover_name: str | None = None,
    hover_data: list[str] | None = None,
    x_axis_title: str = "",
    y_axis_title: str = "",
    trendline: str | None = None,
    max_points: int = 5000,
) -> go.Figure:
    """Scatter plot with optional colour, size encoding and trendline."""
    plot_df = df.dropna(subset=[x, y]).copy()

    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_name=hover_name,
        hover_data=hover_data,
        trendline=trendline,
        title=title,
        color_discrete_sequence=_PALETTE,
        labels={x: x_axis_title or _clean_label(x), y: y_axis_title or _clean_label(y)},
    )

    fig.update_traces(marker=dict(opacity=0.60))

    fig.update_layout(
        **_base_layout(
            title=_subtitle_title(title, size=12),
            xaxis=dict(
                title=x_axis_title or _clean_label(x),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=True,
                gridcolor=_GRID,
                zeroline=False,
                automargin=True,
            ),
            yaxis=dict(
                title=y_axis_title or _clean_label(y),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=True,
                gridcolor=_GRID,
                zeroline=False,
                automargin=True,
            ),
        )
    )

    return fig


# ===========================================================================
# 8. heatmap
# ===========================================================================

def heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    values: str,
    title: str,
    colorscale: str = "Blues",
    x_axis_title: str = "",
    y_axis_title: str = "",
    fmt: str = ".2f",
) -> go.Figure:
    """Heatmap from a tidy long DataFrame."""
    pivot = df.pivot_table(index=y, columns=x, values=values, aggfunc="mean")
    z = pivot.values

    x_labels = [_clean_label(c) for c in pivot.columns.tolist()]
    y_labels = [_wrap_axis_label(_clean_label(r), width=24) for r in pivot.index.tolist()]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            hovertemplate=f"%{{y}} × %{{x}}: %{{z:{fmt}}}<extra></extra>",
            text=[[f"{v:{fmt}}" if not pd.isna(v) else "" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(family=_FONT_FAMILY, size=10, color=_TEXT_DARK),
        )
    )

    fig.update_layout(
        **_base_layout(
            title=_subtitle_title(title, size=12),
            xaxis=dict(
                title=x_axis_title or _clean_label(x),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                tickangle=-30,
                automargin=True,
            ),
            yaxis=dict(
                title=y_axis_title or _clean_label(y),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                automargin=True,
            ),
            margin=dict(l=24, r=24, t=48, b=48),
        )
    )

    return fig


# ===========================================================================
# 9. grouped_bar
# ===========================================================================

def grouped_bar(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    title: str,
    y_labels: dict[str, str] | None = None,
    y_axis_title: str = "",
    x_axis_title: str = "",
    barmode: str = "group",
) -> go.Figure:
    """Grouped or stacked vertical bar chart."""
    y_labels = y_labels or {}
    fig = go.Figure()

    for i, col in enumerate(y_cols):
        display_name = y_labels.get(col, _clean_label(col))
        semantic_color = _semantic_metric_color(display_name, fallback_index=i)

        fig.add_trace(
            go.Bar(
                x=[
                    _wrap_axis_label(_pretty_category_label(v), width=24)
                    if ("category" in str(x).lower() or "product_category" in str(x).lower())
                    else _pretty_payment_label(v) if "payment" in str(x).lower()
                    else _clean_label(v)
                    for v in df[x]
                ],
                y=df[col],
                name=display_name,
                marker_color=semantic_color,
                hovertemplate=f"<b>{display_name}</b><br>%{{x}}<br>%{{y:,.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        **_base_layout(
            title=_subtitle_title(title, size=12),
            barmode=barmode,
            xaxis=dict(
                title=x_axis_title or _clean_label(x),
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=False,
                automargin=True,
            ),
            yaxis=dict(
                title=y_axis_title or "",
                title_font=dict(size=13, color="#0F172A"),
                title_standoff=8,
                tickfont=dict(size=11, color=_TEXT_MEDIUM),
                showgrid=True,
                gridcolor=_GRID,
                zeroline=False,
                automargin=True,
            ),
        )
    )

    return fig


# ===========================================================================
# 10. kpi_delta_card
# ===========================================================================

def kpi_delta_card(
    label: str,
    value: float | int | str,
    delta: float | None = None,
    fmt: str = "{:,.0f}",
    delta_fmt: str = "{:+.2f}",
    delta_color: str = "normal",
) -> dict:
    """Return a dict suitable for unpacking into st.metric(...)."""
    display_value = fmt.format(value) if isinstance(value, (int, float)) else str(value)
    display_delta = delta_fmt.format(delta) if delta is not None else None

    return dict(
        label=label,
        value=display_value,
        delta=display_delta,
        delta_color=delta_color,
    )


# ### `ml_module(1).py` — Delivery-delay machine-learning module

# In[ ]:


"""
Olist E-Commerce Delivery Delay Prediction ML Module
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DATA_PATH = PROJECT_ROOT / "data" / "ml_delay_dataset.csv"

TARGET = "is_delayed"
TIMESTAMP = "order_purchase_timestamp"

TRAIN_CUTOFF = pd.Timestamp("2018-02-01")
VALIDATION_CUTOFF = pd.Timestamp("2018-05-26 18:17:10.2")

CATEGORICAL_FEATURES = [
    "primary_payment_type",
    "product_category_name_english",
    "customer_state",
    "seller_state",
]

DROP_FEATURES = [
    TARGET,
    TIMESTAMP,
]


# ============================================================
# Data loading
# ============================================================

def load_ml_data(path=ML_DATA_PATH):
    """Load the leakage-safe ML dataset."""

    df = pd.read_csv(
        path,
        parse_dates=[TIMESTAMP],
    )

    return df


# ============================================================
# Dataset validation
# ============================================================

def validate_ml_dataset(df):
    """Validate target and leakage constraints."""

    forbidden = {
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "delivery_time_days",
        "delay_days",
        "review_score",
        "review_creation_date",
        "review_answer_timestamp",
    }

    found = forbidden.intersection(df.columns)

    if found:
        raise ValueError(
            f"Potential leakage columns detected: {sorted(found)}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    target_values = set(df[TARGET].unique())

    if not target_values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Target must contain only 0/1. Found: {target_values}"
        )

    if df[TIMESTAMP].isna().any():
        raise ValueError(
            f"{TIMESTAMP} contains missing values."
        )

    return True


# ============================================================
# Temporal split
# ============================================================

def temporal_split(
    df,
    train_cutoff=TRAIN_CUTOFF,
    validation_cutoff=VALIDATION_CUTOFF,
):
    """
    Chronological train / validation / test split.
    """

    train = df[
        df[TIMESTAMP] < train_cutoff
    ].copy()

    validation = df[
        (df[TIMESTAMP] >= train_cutoff)
        & (df[TIMESTAMP] < validation_cutoff)
    ].copy()

    test = df[
        df[TIMESTAMP] >= validation_cutoff
    ].copy()

    if train.empty:
        raise ValueError("Training split is empty.")

    if validation.empty:
        raise ValueError("Validation split is empty.")

    if test.empty:
        raise ValueError("Test split is empty.")

    return train, validation, test


# ============================================================
# Feature preparation
# ============================================================

def prepare_xy(df):
    """Separate features and target."""

    X = df.drop(
        columns=DROP_FEATURES,
        errors="ignore",
    ).copy()

    y = df[TARGET].astype(int).copy()

    return X, y


def get_feature_columns(X):
    """Separate numeric and categorical features."""

    categorical = [
        c
        for c in CATEGORICAL_FEATURES
        if c in X.columns
    ]

    numeric = [
        c
        for c in X.columns
        if c not in categorical
    ]

    return numeric, categorical


# ============================================================
# Preprocessing
# ============================================================

def build_preprocessor(X):
    """Create preprocessing pipeline."""

    numeric_features, categorical_features = (
        get_feature_columns(X)
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


# ============================================================
# Logistic Regression
# ============================================================

def build_logistic_model(X):
    """Build Logistic Regression baseline."""

    preprocessor = build_preprocessor(X)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# ============================================================
# Random Forest
# ============================================================

def build_random_forest_model(X):
    """Build validated Random Forest model."""

    preprocessor = build_preprocessor(X)

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# ============================================================
# Training
# ============================================================

def train_model(model, train_df):
    """Train model using training data only."""

    X_train, y_train = prepare_xy(train_df)

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# Prediction
# ============================================================

def predict_probability(model, df):
    """Return probability of delivery delay."""

    X = df.drop(
        columns=DROP_FEATURES,
        errors="ignore",
    ).copy()

    return model.predict_proba(X)[:, 1]


def predict_class(model, df, threshold=0.5):
    """Convert probability into binary prediction."""

    probability = predict_probability(
        model,
        df,
    )

    return (
        probability >= threshold
    ).astype(int)


# ============================================================
# Model evaluation
# ============================================================

def evaluate_model(
    model,
    df,
    threshold=0.5,
):
    """Calculate classification metrics."""

    X, y = prepare_xy(df)

    probability = model.predict_proba(X)[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                prediction,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probability,
            )
        ),
    }


# ============================================================
# Threshold analysis
# ============================================================

def threshold_analysis(
    model,
    df,
    thresholds=None,
):
    """Evaluate precision, recall and F1 at different thresholds."""

    if thresholds is None:
        thresholds = [
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
        ]

    X, y = prepare_xy(df)

    probability = model.predict_proba(X)[:, 1]

    results = []

    for threshold in thresholds:

        prediction = (
            probability >= threshold
        ).astype(int)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y,
                    prediction,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    prediction,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y,
                    prediction,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(results)


# ============================================================
# Random Forest feature importance
# ============================================================

def get_random_forest_importance(
    model,
    top_n=25,
):
    """Return top Random Forest feature importances."""

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = pd.Series(
        classifier.feature_importances_,
        index=feature_names,
        name="importance",
    )

    result = (
        importance
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    result.columns = [
        "feature",
        "importance",
    ]

    return result


# ============================================================
# Train both baseline models
# ============================================================

def train_baseline_models():
    """Train Logistic Regression and Random Forest."""

    df = load_ml_data()

    validate_ml_dataset(df)

    train, validation, test = temporal_split(df)

    X_train, _ = prepare_xy(train)

    logistic = build_logistic_model(
        X_train
    )

    random_forest = build_random_forest_model(
        X_train
    )

    logistic = train_model(
        logistic,
        train,
    )

    random_forest = train_model(
        random_forest,
        train,
    )

    return {
        "logistic": logistic,
        "random_forest": random_forest,
        "train": train,
        "validation": validation,
        "test": test,
    }


# ### `01_overview(1).py` — Overview dashboard page

# In[ ]:


"""
pages/01_overview.py
====================
Executive Overview page — high-level KPI scorecards, monthly order &
revenue trends, and order status breakdown.

Data grain: order-level master only (one row per order).
No item-level data is used on this page.

Business questions answered
---------------------------
- How do orders and revenue change month-over-month?
- What is the on-time delivery rate and average review score?
- How are orders distributed across statuses?
- What is the average and median order value?
- How many unique customers placed orders in the selected period?
"""

import streamlit as st

from src.data_loader import load_master, load_item_master
from src.filters import render_sidebar_filters, apply_master_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load data (cached at app level; re-used across pages)
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
fs: FilterState = render_sidebar_filters(master, items)

# Filtered slices
# Default filter = delivered orders, Jan 2017 – Oct 2018, no 2016 partial.
master_f = apply_master_filters(master, fs)

# All-status slice for cancellation rate and status breakdown
fs_all = FilterState(
    date_start=fs.date_start,
    date_end=fs.date_end,
    include_2016=fs.include_2016,
    order_statuses=[],          # no status filter
    customer_states=fs.customer_states,
)
master_all = apply_master_filters(master, fs_all)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("📊 Executive Overview")
st.caption(
    "High-level KPIs, monthly order & revenue trends, and order status breakdown. "
    "Data grain: one row per order. Default view: delivered orders, Jan 2017 – Oct 2018."
)

if len(master_f) == 0:
    st.warning("No orders match the current filters. Adjust the sidebar selections.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI scorecards — row 1: volume & value
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

n_orders     = kpis.total_orders(master_f)
sales_val    = kpis.total_sales_value(master_f)
freight_val  = kpis.total_freight_value(master_f)
aov          = kpis.avg_order_value(master_f)
median_ov    = kpis.median_order_value(master_f)
n_customers  = kpis.unique_customers(master_f)
cancel_rate  = kpis.cancellation_rate(master_all)

col1, col2, col3, col4 = st.columns(4)
col1.metric(**charts.kpi_delta_card("Total Orders",       n_orders,    fmt="{:,.0f}"))
col2.metric(**charts.kpi_delta_card("Total Sales Value",  sales_val,   fmt="R$ {:,.0f}"))
col3.metric(**charts.kpi_delta_card("Total Freight",      freight_val, fmt="R$ {:,.0f}"))
col4.metric(**charts.kpi_delta_card("Avg Order Value",    aov,         fmt="R$ {:,.2f}"))

col5, col6, col7, col8 = st.columns(4)
col5.metric(**charts.kpi_delta_card("Median Order Value", median_ov,   fmt="R$ {:,.2f}"))
col6.metric(**charts.kpi_delta_card("Unique Customers",   n_customers, fmt="{:,.0f}"))
col7.metric(**charts.kpi_delta_card("Cancellation Rate",  cancel_rate * 100,
                                    fmt="{:.2f}%",
                                    delta_color="inverse"))

# ---------------------------------------------------------------------------
# KPI scorecards — row 2: delivery & reviews (delivered orders only)
# ---------------------------------------------------------------------------
st.divider()

otr         = kpis.on_time_rate(master_f)
dr          = kpis.delay_rate(master_f)
avg_del     = kpis.avg_delivery_days(master_f)
med_del     = kpis.median_delivery_days(master_f)
avg_late    = kpis.avg_late_delay_days(master_f)
avg_review  = kpis.avg_review_score(master_f)
five_star   = kpis.five_star_rate(master_f)
low_score   = kpis.low_score_rate(master_f)
rev_cov     = kpis.review_coverage(master_f)

col9, col10, col11, col12 = st.columns(4)
col9.metric(**charts.kpi_delta_card("On-Time Delivery",   otr * 100,   fmt="{:.1f}%"))
col10.metric(**charts.kpi_delta_card("Delay Rate",        dr * 100,    fmt="{:.1f}%",
                                     delta_color="inverse"))
col11.metric(**charts.kpi_delta_card("Avg Delivery Days", avg_del,     fmt="{:.1f} days"))
col12.metric(**charts.kpi_delta_card("Median Delivery",   med_del,     fmt="{:.1f} days"))

col13, col14, col15, col16 = st.columns(4)
col13.metric(**charts.kpi_delta_card("Avg Delay (late)",  avg_late,    fmt="{:.1f} days",
                                     delta_color="inverse"))
col14.metric(**charts.kpi_delta_card("Avg Review Score",  avg_review,  fmt="{:.2f} / 5"))
col15.metric(**charts.kpi_delta_card("5-Star Rate",       five_star * 100, fmt="{:.1f}%"))
col16.metric(**charts.kpi_delta_card("1–2 Star Rate",     low_score * 100, fmt="{:.1f}%",
                                     delta_color="inverse"))

# Delivery denominator note
n_with_delivery = int(master_f["delivery_time_days"].notna().sum())
n_with_review   = int(master_f["review_score"].notna().sum())
st.caption(
    f"Delivery KPIs: {n_with_delivery:,} orders with resolved delivery date. "
    f"Review KPIs: {n_with_review:,} orders with a review score "
    f"({rev_cov * 100:.1f}% coverage). "
    "Avg Delay shows late orders (is_delayed = 1) only."
)

# ---------------------------------------------------------------------------
# Monthly trend — orders & sales (dual-axis)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Monthly Order Volume & Sales Value")

monthly = kpis.monthly_orders_and_sales(master_f)

if len(monthly) > 0:
    # 2016 partial-period annotation flag
    has_2016 = fs.include_2016 and any(
        str(m).startswith("2016") for m in monthly["order_month_str"]
    )

    fig_trend = charts.line_trend(
        monthly,
        x="order_month_str",
        y_cols=["order_count", "total_sales"],
        title="Monthly Orders & Sales Value",
        y_labels={"order_count": "Orders", "total_sales": "Sales Value (R$)"},
        secondary_y_col="total_sales",
        y_axis_title="Orders",
        y2_axis_title="Sales Value (R$)",
    )

    if has_2016:
        fig_trend.add_annotation(
            text="◀ 2016 partial period",
            x=monthly["order_month_str"].iloc[0],
            y=monthly["order_count"].iloc[0],
            showarrow=True, arrowhead=2,
            font=dict(size=11, color="#57606a"),
            ax=60, ay=-30,
        )

    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No monthly data available for the selected period.")

# ---------------------------------------------------------------------------
# Monthly AOV trend
# ---------------------------------------------------------------------------
st.subheader("Monthly Average Order Value")

fig_aov = charts.line_trend(
    monthly,
    x="order_month_str",
    y_cols=["avg_order_value"],
    title="Monthly Average Order Value (R$)",
    y_labels={"avg_order_value": "AOV (R$)"},
    y_axis_title="R$",
)
st.plotly_chart(fig_aov, use_container_width=True)

# ---------------------------------------------------------------------------
# Order status breakdown
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Order Status Distribution")
st.caption(
    "Shows all order statuses in the selected date range, regardless of the "
    "status filter applied to delivery and value KPIs above."
)

obs = kpis.orders_by_status(master_all)

col_donut, col_table = st.columns([1, 1])
with col_donut:
    fig_status = charts.donut_chart(
        labels=obs["order_status"].tolist(),
        values=obs["order_count"].tolist(),
        title="Order Status Share",
    )
    st.plotly_chart(fig_status, use_container_width=True)

with col_table:
    st.markdown("**Order counts by status**")
    display_obs = obs.copy()
    display_obs["share_pct"] = display_obs["share_pct"].map("{:.2f}%".format)
    display_obs.columns = ["Status", "Orders", "Share"]
    st.dataframe(display_obs, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Dataset / filter transparency footer
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes & filter summary", expanded=False):
    st.markdown(
        f"""
        **Active filters**
        - Date range: {fs.date_start} → {fs.date_end}
        - 2016 partial period: {"included" if fs.include_2016 else "excluded"}
        - Order statuses (KPI rows): {", ".join(fs.order_statuses) if fs.order_statuses else "all"}
        - Customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}

        **Denominator rules**
        - Total Orders, Sales, Customers: all rows matching the status filter.
        - Delivery KPIs: orders with a non-null `delivery_time_days` ({n_with_delivery:,} rows).
        - Delay KPIs: orders with a non-null `is_delayed` flag.
        - Avg Delay (late): orders where `is_delayed == 1` ({int((master_f["is_delayed"] == 1).sum()):,} rows).
        - Review KPIs: orders with a non-null `review_score` ({n_with_review:,} rows).
        - Cancellation Rate: {int((master_all["order_status"] == "canceled").sum()):,} cancelled /
          {len(master_all):,} total orders in period.

        **Data grain**: this page uses the order-level master only (one row per order).
        No item-level data is used. No double-counting of orders, payments, or reviews.
        """
    )


# ### `02_products_categories(1).py` — Products and categories page

# In[ ]:


"""
pages/02_products_categories.py
================================
Products & Categories page.

Data grain
----------
- Sales / freight / item KPIs  : item-level master  (one row per order-item)
- Category delivery / reviews  : build_order_category_table() → one row per order
- No master (order-grain) data is used here — all figures come from the item grain.

Business questions answered
---------------------------
- Which categories generate the highest sales value and item volume?
- What is the typical item price and freight cost per category?
- How many sellers operate in each category?
- How do category sales trend over time?
"""

import streamlit as st

from src.data_loader import load_master, load_item_master, build_order_category_table
from src.filters import render_sidebar_filters, apply_item_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)
items_f = apply_item_filters(items, fs)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("📦 Products & Categories")
st.caption(
    "All figures use item-level data (one row per order-item). "
    "Sales value = SUM(item price). No order-level totals are used here."
)

if len(items_f) == 0:
    st.warning("No items match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Tier-1 KPI scorecards
# ---------------------------------------------------------------------------
sbc = kpis.sales_by_category(items_f)

total_items   = int(items_f["order_item_id"].count())
total_cat_sales = float(sbc["total_sales_value"].sum())
avg_price     = float(items_f["price"].mean())
avg_freight   = float(items_f["freight_value"].mean())
n_categories  = int(items_f["product_category_name_english"].nunique())
n_sellers     = int(items_f["seller_id"].nunique())

st.subheader("Key Performance Indicators")
c1, c2, c3 = st.columns(3)
c1.metric(**charts.kpi_delta_card("Items Sold",           total_items,     fmt="{:,.0f}"))
c2.metric(**charts.kpi_delta_card("Total Sales Value",    total_cat_sales, fmt="R$ {:,.0f}"))
c3.metric(**charts.kpi_delta_card("Avg Item Price",       avg_price,       fmt="R$ {:,.2f}"))

c4, c5, c6 = st.columns(3)
c4.metric(**charts.kpi_delta_card("Avg Freight / Item",   avg_freight,     fmt="R$ {:,.2f}"))
c5.metric(**charts.kpi_delta_card("Active Categories",    n_categories,    fmt="{:,.0f}"))
c6.metric(**charts.kpi_delta_card("Active Sellers",       n_sellers,       fmt="{:,.0f}"))

st.caption(
    "Sales Value = SUM(item price) from the item-level master. "
    "Unique orders: "
    f"{items_f['order_id'].nunique():,}."
)

# ---------------------------------------------------------------------------
# Top-N control
# ---------------------------------------------------------------------------
st.divider()
top_n = st.slider("Number of categories to display", min_value=5, max_value=72,
                  value=15, step=5, key="cat_top_n")

# ---------------------------------------------------------------------------
# Sales value by category
# ---------------------------------------------------------------------------
st.subheader("Sales Value by Category")
fig_sales = charts.bar_horizontal(
    sbc, x="total_sales_value", y="product_category_name_english",
    title=f"Top {top_n} Categories — Total Sales Value (R$)",
    top_n=top_n, x_axis_title="Sales Value (R$)",
)
st.plotly_chart(fig_sales, use_container_width=True)

# ---------------------------------------------------------------------------
# Items sold by category
# ---------------------------------------------------------------------------
st.subheader("Items Sold by Category")
fig_items = charts.bar_horizontal(
    sbc, x="item_count", y="product_category_name_english",
    title=f"Top {top_n} Categories — Items Sold",
    top_n=top_n, x_axis_title="Items Sold",
)
st.plotly_chart(fig_items, use_container_width=True)

# ---------------------------------------------------------------------------
# Avg item price by category
# ---------------------------------------------------------------------------
st.subheader("Average Item Price by Category")
fig_price = charts.bar_horizontal(
    sbc, x="avg_item_price", y="product_category_name_english",
    title=f"Top {top_n} Categories — Avg Item Price (R$)",
    top_n=top_n, x_axis_title="Avg Item Price (R$)",
)
st.plotly_chart(fig_price, use_container_width=True)

# ---------------------------------------------------------------------------
# Avg freight per item by category
# ---------------------------------------------------------------------------
st.subheader("Avg Freight per Item by Category")
fig_freight = charts.bar_horizontal(
    sbc, x="avg_freight_per_item", y="product_category_name_english",
    title=f"Top {top_n} Categories — Avg Freight per Item (R$)",
    top_n=top_n, x_axis_title="Avg Freight (R$)",
)
st.plotly_chart(fig_freight, use_container_width=True)

# ---------------------------------------------------------------------------
# Monthly item sales trend (filtered items)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Monthly Item Sales Trend")
ibm = kpis.items_by_month(items_f)
fig_trend = charts.line_trend(
    ibm, x="order_month_str",
    y_cols=["item_count", "total_item_sales_value"],
    title="Monthly Items Sold & Sales Value",
    y_labels={"item_count": "Items Sold",
               "total_item_sales_value": "Sales Value (R$)"},
    secondary_y_col="total_item_sales_value",
    y_axis_title="Items Sold",
    y2_axis_title="Sales Value (R$)",
)
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------------------------------------------------------
# Price vs volume scatter (category level)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Category Price vs Volume")
st.caption(
    "Bubble size = number of unique sellers in the category. "
    "Top categories by sales value are labelled."
)
fig_scatter = charts.scatter_plot(
    sbc,
    x="avg_item_price",
    y="item_count",
    title="Avg Item Price vs Items Sold (by Category)",
    size="unique_sellers",
    hover_name="product_category_name_english",
    hover_data=["total_sales_value", "unique_sellers"],
    x_axis_title="Avg Item Price (R$)",
    y_axis_title="Items Sold",
    max_points=len(sbc),   # all categories — cardinality is small (≤ 72)
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Category detail table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Category Detail Table")
display_sbc = sbc.rename(columns={
    "product_category_name_english": "Category",
    "item_count":        "Items Sold",
    "unique_orders":     "Unique Orders",
    "unique_sellers":    "Sellers",
    "total_sales_value": "Sales Value (R$)",
    "avg_item_price":    "Avg Price (R$)",
    "median_item_price": "Median Price (R$)",
    "avg_freight_per_item": "Avg Freight (R$)",
    "total_freight_value":  "Total Freight (R$)",
}).copy()
for col in ["Sales Value (R$)", "Avg Price (R$)", "Median Price (R$)",
            "Avg Freight (R$)", "Total Freight (R$)"]:
    display_sbc[col] = display_sbc[col].map("R$ {:,.2f}".format)

st.dataframe(display_sbc, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    st.markdown(
        f"""
        **Grain**: item-level master (one row per order-item).
        **Sales value** = `SUM(items.price)`, not `SUM(master.total_price)`.
        Category attribution is at item level — each item contributes its own price
        to its own category.

        **Active filters**: {fs.date_start} → {fs.date_end} • 
        statuses: {", ".join(fs.order_statuses) if fs.order_statuses else "all"} • 
        categories: {", ".join(fs.categories) if fs.categories else "all"} • 
        seller states: {", ".join(fs.seller_states) if fs.seller_states else "all"}

        **Rows in view**: {len(items_f):,} items • 
        {items_f['order_id'].nunique():,} unique orders • 
        {n_sellers:,} sellers • {n_categories:,} categories
        """
    )


# ### `03_delivery_operations(1).py` — Delivery operations page

# In[ ]:


"""
pages/03_delivery_operations.py
================================
Delivery & Operations page.

Data grain
----------
- Overall KPIs, state analysis, monthly trend : order-level master (delivered)
- Category-level delivery                     : build_order_category_table()
  → one row per order; category assigned via mode with alpha tie-break.

Key rules
---------
- All delivery KPIs exclude rows where delivery_time_days is null.
- delay_days sign: negative = arrived early.
  "Avg Delay" KPI uses only is_delayed == 1 rows.
- Category delivery denominator = unique orders (not item rows).
- Multi-category orders attributed to primary category (documented in UI).
"""

import streamlit as st

from src.data_loader import load_master, load_item_master, build_order_category_table
from src.filters import render_sidebar_filters, apply_master_filters, apply_item_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)
master_f = apply_master_filters(master, fs)
items_f  = apply_item_filters(items, fs)

# For cancellation rate: all statuses
fs_all   = FilterState(
    date_start=fs.date_start, date_end=fs.date_end,
    include_2016=fs.include_2016, order_statuses=[],
    customer_states=fs.customer_states,
)
master_all = apply_master_filters(master, fs_all)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🚚 Delivery & Operations")
st.caption(
    "Overall KPIs and state-level analysis use the order-level master (one row per order). "
    "Category-level delivery uses an order-level category attribution table — each order "
    "counted once."
)

if len(master_f) == 0:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Overall KPI scorecards
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

avg_del   = kpis.avg_delivery_days(master_f)
med_del   = kpis.median_delivery_days(master_f)
otr       = kpis.on_time_rate(master_f)
dr        = kpis.delay_rate(master_f)
avg_late  = kpis.avg_late_delay_days(master_f)
cancel_r  = kpis.cancellation_rate(master_all)

n_del     = int(master_f["delivery_time_days"].notna().sum())
n_delayed = int((master_f["is_delayed"] == 1).sum())

c1, c2, c3 = st.columns(3)
c1.metric(**charts.kpi_delta_card("Avg Delivery Time",   avg_del,          fmt="{:.1f} days"))
c2.metric(**charts.kpi_delta_card("Median Delivery Time", med_del,         fmt="{:.1f} days"))
c3.metric(**charts.kpi_delta_card("On-Time Delivery Rate", otr * 100,      fmt="{:.1f}%"))

c4, c5, c6 = st.columns(3)
c4.metric(**charts.kpi_delta_card("Delay Rate",          dr * 100,         fmt="{:.1f}%",
                                   delta_color="inverse"))
c5.metric(**charts.kpi_delta_card("Avg Delay (late orders)", avg_late,     fmt="{:.1f} days",
                                   delta_color="inverse"))
c6.metric(**charts.kpi_delta_card("Cancellation Rate",   cancel_r * 100,   fmt="{:.2f}%",
                                   delta_color="inverse"))

st.caption(
    f"Delivery KPIs: {n_del:,} orders with resolved delivery date. "
    f"Avg Delay: {n_delayed:,} late orders (is_delayed = 1). "
    "Negative delay_days = arrived early — excluded from Avg Delay."
)

# ---------------------------------------------------------------------------
# Delivery time distribution
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Delivery Time Distribution")

fig_hist = charts.histogram(
    master_f["delivery_time_days"].dropna(),
    title="Delivery Time Distribution",
    xaxis_title="Days from purchase to delivery",
    nbins=40,
    clip_upper=60,
    add_delay_note=False,
)
st.plotly_chart(fig_hist, use_container_width=True)
st.caption("Values above 60 days clipped for readability. See table below for full range.")

# On-time vs late donut
col_donut, col_stats = st.columns([1, 1])
with col_donut:
    fig_donut = charts.donut_chart(
        labels=["On-Time", "Late"],
        values=[int((master_f["is_delayed"] == 0).sum()),
                int((master_f["is_delayed"] == 1).sum())],
        title="On-Time vs Late Orders",
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_stats:
    st.markdown("**Delivery time summary**")
    del_series = master_f["delivery_time_days"].dropna()
    st.dataframe(
        del_series.describe().rename({
            "count": "Orders", "mean": "Mean (days)", "std": "Std Dev",
            "min": "Min", "25%": "25th pct", "50%": "Median",
            "75%": "75th pct", "max": "Max",
        }).to_frame("Value").reset_index().rename(columns={"index": "Metric"}),
        hide_index=True, use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Monthly delivery trend
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Monthly Delivery Performance")

dbm = kpis.delivery_by_month(master_f)

fig_monthly = charts.line_trend(
    dbm,
    x="order_month_str",
    y_cols=["avg_delivery_days", "delay_rate"],
    title="Monthly Avg Delivery Days & Delay Rate",
    y_labels={"avg_delivery_days": "Avg Delivery (days)",
               "delay_rate": "Delay Rate"},
    secondary_y_col="delay_rate",
    y_axis_title="Avg Delivery (days)",
    y2_axis_title="Delay Rate",
    add_delay_note=True,
)
st.plotly_chart(fig_monthly, use_container_width=True)

# ---------------------------------------------------------------------------
# Delivery by customer state
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Delivery Performance by Customer State")

dbs = kpis.delivery_by_state(master_f)

tab_del, tab_delay = st.tabs(["Avg Delivery Days", "Delay Rate"])
with tab_del:
    fig_state_del = charts.bar_horizontal(
        dbs, x="avg_delivery_days", y="customer_state",
        title="Avg Delivery Days by Customer State",
        top_n=27, x_axis_title="Avg Delivery Days",
    )
    st.plotly_chart(fig_state_del, use_container_width=True)

with tab_delay:
    # Convert delay_rate to percentage for display
    dbs_pct = dbs.copy()
    dbs_pct["delay_rate_pct"] = dbs_pct["delay_rate"] * 100
    fig_state_dr = charts.bar_horizontal(
        dbs_pct, x="delay_rate_pct", y="customer_state",
        title="Delay Rate (%) by Customer State",
        top_n=27, x_axis_title="Delay Rate (%)",
        add_multicategory_note=False,
    )
    st.plotly_chart(fig_state_dr, use_container_width=True)

with st.expander("State delivery detail table"):
    display_dbs = dbs.copy()
    display_dbs["delay_rate"] = display_dbs["delay_rate"].map("{:.1%}".format)
    display_dbs["avg_delivery_days"]    = display_dbs["avg_delivery_days"].map("{:.1f}".format)
    display_dbs["median_delivery_days"] = display_dbs["median_delivery_days"].map("{:.1f}".format)
    display_dbs = display_dbs.rename(columns={
        "customer_state": "State", "order_count": "Orders",
        "avg_delivery_days": "Avg Days", "median_delivery_days": "Median Days",
        "delay_rate": "Delay Rate", "n_delayed": "Late", "n_on_time": "On-Time",
    })
    st.dataframe(display_dbs, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Category-level delivery  (order-category table — one row per order)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Delivery Performance by Category")
st.caption(
    "Each order is counted once, attributed to its most frequently purchased category. "
    "Orders with items from multiple categories are attributed to the dominant category."
)

oct_df = build_order_category_table(items_f)
dbc    = kpis.delivery_by_category(oct_df)

top_n_cat = st.slider("Categories to display", 5, min(72, len(dbc)), 20,
                       step=5, key="del_cat_top_n")

tab_cat_del, tab_cat_dr = st.tabs(["Avg Delivery Days", "Delay Rate"])
with tab_cat_del:
    fig_cat_del = charts.bar_horizontal(
        dbc, x="avg_delivery_days", y="primary_category",
        title=f"Top {top_n_cat} Categories — Avg Delivery Days",
        top_n=top_n_cat, x_axis_title="Avg Delivery Days",
        add_multicategory_note=True,
    )
    st.plotly_chart(fig_cat_del, use_container_width=True)

with tab_cat_dr:
    dbc_pct = dbc.copy()
    dbc_pct["delay_rate_pct"] = dbc_pct["delay_rate"] * 100
    fig_cat_dr = charts.bar_horizontal(
        dbc_pct, x="delay_rate_pct", y="primary_category",
        title=f"Top {top_n_cat} Categories — Delay Rate (%)",
        top_n=top_n_cat, x_axis_title="Delay Rate (%)",
        add_multicategory_note=True,
    )
    st.plotly_chart(fig_cat_dr, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    st.markdown(
        f"""
        **Delivery KPI denominator**: {n_del:,} orders with a resolved
        `delivery_time_days` (orders without a delivery date are excluded).

        **Delay sign convention**: `delay_days` is negative when an order
        arrives *before* the estimated date. The Avg Delay KPI uses only
        orders flagged `is_delayed = 1` ({n_delayed:,} orders).

        **Category delivery**: built from `build_order_category_table()` —
        one row per order. {int(oct_df['has_multiple_categories'].sum()):,} orders
        contained items from multiple categories and were attributed to the
        most frequently purchased category.

        **Active filters**: {fs.date_start} → {fs.date_end} •
        statuses: {", ".join(fs.order_statuses) if fs.order_statuses else "all"} •
        customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}
        """
    )


# ### `04_customer_experience(1).py` — Customer experience page

# In[ ]:


"""
pages/04_customer_experience.py
================================
Customer Experience & Reviews page.

Data grain
----------
- Review score KPIs, distribution, monthly trend,
  review vs delay, state-level reviews : order-level master (one row per order)
- Category-level reviews               : build_order_category_table()
  → one row per order, primary_category assigned by mode with alpha tie-break

Key rules
---------
- Review KPIs exclude rows with null review_score; denominator shown in UI.
- Review coverage = reviews with score / total orders.
- Association between delay and review is shown without causal attribution.
- 5-star rate and 1-2 star rate use non-null review_score as denominator.
"""

import streamlit as st

from src.data_loader import load_master, load_item_master, build_order_category_table
from src.filters import render_sidebar_filters, apply_master_filters, apply_item_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)
master_f = apply_master_filters(master, fs)
items_f  = apply_item_filters(items, fs)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("⭐ Customer Experience")
st.caption(
    "Review KPIs use the order-level master (one row per order). "
    "Category review scores use an order-level attribution table — each order counted once."
)

if len(master_f) == 0:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI scorecards
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

avg_rev  = kpis.avg_review_score(master_f)
five_s   = kpis.five_star_rate(master_f)
low_s    = kpis.low_score_rate(master_f)
rev_cov  = kpis.review_coverage(master_f)
n_reviews = int(master_f["review_score"].notna().sum())
n_orders  = kpis.total_orders(master_f)

c1, c2, c3, c4 = st.columns(4)
c1.metric(**charts.kpi_delta_card("Avg Review Score",  avg_rev,       fmt="{:.2f} / 5"))
c2.metric(**charts.kpi_delta_card("5-Star Rate",       five_s * 100,  fmt="{:.1f}%"))
c3.metric(**charts.kpi_delta_card("1–2 Star Rate",     low_s * 100,   fmt="{:.1f}%",
                                   delta_color="inverse"))
c4.metric(**charts.kpi_delta_card("Review Coverage",   rev_cov * 100, fmt="{:.1f}%"))

st.caption(
    f"Review KPIs: {n_reviews:,} orders with a review score "
    f"({rev_cov*100:.1f}% of {n_orders:,} orders in period)."
)

# ---------------------------------------------------------------------------
# Review score distribution
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Score Distribution")

rsd = kpis.review_score_distribution(master_f)
rsd["label"] = rsd["review_score"].astype(str) + " ★  " + rsd["share_pct"].map("({:.1f}%)".format)

fig_dist = charts.bar_vertical(
    rsd, x="review_score", y="count",
    title="Review Score Distribution",
    x_axis_title="Review Score", y_axis_title="Orders",
    text_col="label",
    color_discrete_sequence=["#3b82d4"],
)
st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------------------------
# Monthly avg review trend
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Monthly Average Review Score")

monthly_rev = (
    master_f.dropna(subset=["review_score"])
    .groupby("order_month_str", sort=True)
    .agg(avg_review_score=("review_score", "mean"),
         n_reviews=("review_score", "count"))
    .reset_index()
)

if len(monthly_rev) > 0:
    fig_rev_trend = charts.line_trend(
        monthly_rev,
        x="order_month_str",
        y_cols=["avg_review_score"],
        title="Monthly Avg Review Score",
        y_labels={"avg_review_score": "Avg Review Score"},
        y_axis_title="Avg Score",
    )
    st.plotly_chart(fig_rev_trend, use_container_width=True)

# ---------------------------------------------------------------------------
# Review score vs delivery delay (association — no causation claim)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Score: On-Time vs Delayed Orders")
st.caption(
    "⚠️ This chart shows an **association** between delivery status and review scores. "
    "It does not establish a causal relationship."
)

rvd = kpis.review_vs_delay(master_f)

if len(rvd) == 2:
    rvd_display = rvd.copy()
    rvd_display["delivery_status"] = rvd_display["is_delayed"].map(
        {0: "On-Time", 1: "Delayed"}
    ).fillna(rvd_display["is_delayed"].astype(str))

    col_bar, col_table = st.columns([2, 1])
    with col_bar:
        fig_rvd = charts.grouped_bar(
            rvd_display,
            x="delivery_status",
            y_cols=["avg_review_score"],
            title="Avg Review Score by Delivery Status",
            y_labels={"avg_review_score": "Avg Review Score"},
            y_axis_title="Avg Review Score",
        )
        st.plotly_chart(fig_rvd, use_container_width=True)

    with col_table:
        st.markdown("**Summary**")
        tbl = rvd_display[["delivery_status", "order_count",
                            "avg_review_score", "low_score_rate", "five_star_rate"]].copy()
        tbl["avg_review_score"] = tbl["avg_review_score"].map("{:.2f}".format)
        tbl["low_score_rate"]   = tbl["low_score_rate"].map("{:.1%}".format)
        tbl["five_star_rate"]   = tbl["five_star_rate"].map("{:.1%}".format)
        tbl = tbl.rename(columns={
            "delivery_status": "Status", "order_count": "Orders",
            "avg_review_score": "Avg Score", "low_score_rate": "1-2★ Rate",
            "five_star_rate": "5★ Rate",
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)
else:
    st.info("Delay data not available for the current filter selection.")

# ---------------------------------------------------------------------------
# Review score by customer state
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Score by Customer State")

rbs = kpis.review_by_state(master_f)

tab_avg, tab_low = st.tabs(["Avg Review Score", "1–2 Star Rate"])
with tab_avg:
    fig_state_rev = charts.bar_horizontal(
        rbs, x="avg_review_score", y="customer_state",
        title="Avg Review Score by Customer State (worst first)",
        top_n=27, x_axis_title="Avg Review Score",
    )
    st.plotly_chart(fig_state_rev, use_container_width=True)
with tab_low:
    rbs_pct = rbs.copy()
    rbs_pct["low_score_pct"] = rbs_pct["low_score_rate"] * 100
    fig_state_low = charts.bar_horizontal(
        rbs_pct, x="low_score_pct", y="customer_state",
        title="1–2 Star Rate (%) by Customer State",
        top_n=27, x_axis_title="1–2 Star Rate (%)",
    )
    st.plotly_chart(fig_state_low, use_container_width=True)

# ---------------------------------------------------------------------------
# Category-level review scores  (order-category table)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Score by Category")
st.caption(
    "Each order is counted once, attributed to its most frequently purchased category. "
    "Multi-category orders attributed to dominant category."
)

oct_df = build_order_category_table(items_f)
rbc    = kpis.review_by_category(oct_df)

top_n_cat = st.slider("Categories to display", 5, min(72, len(rbc)), 20,
                       step=5, key="rev_cat_top_n")

tab_cat_avg, tab_cat_low = st.tabs(["Avg Review Score", "1–2 Star Rate"])
with tab_cat_avg:
    fig_cat_avg = charts.bar_horizontal(
        rbc, x="avg_review_score", y="primary_category",
        title=f"Bottom {top_n_cat} Categories — Avg Review Score",
        top_n=top_n_cat, x_axis_title="Avg Review Score",
        add_multicategory_note=True,
    )
    st.plotly_chart(fig_cat_avg, use_container_width=True)
with tab_cat_low:
    rbc_pct = rbc.copy()
    rbc_pct["low_score_pct"] = rbc_pct["low_score_rate"] * 100
    fig_cat_low = charts.bar_horizontal(
        rbc_pct, x="low_score_pct", y="primary_category",
        title=f"Top {top_n_cat} Categories — 1–2 Star Rate (%)",
        top_n=top_n_cat, x_axis_title="1–2 Star Rate (%)",
        add_multicategory_note=True,
    )
    st.plotly_chart(fig_cat_low, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    st.markdown(
        f"""
        **Review KPI denominator**: {n_reviews:,} orders with a non-null `review_score`
        out of {n_orders:,} total orders ({rev_cov*100:.1f}% coverage).

        **Delay association**: Orders with delayed delivery show a different review score
        distribution. This is an **association**, not a proven causal relationship —
        other factors may also influence review scores.

        **Category reviews**: built from `build_order_category_table()` — one row per order.
        {int(oct_df["has_multiple_categories"].sum()):,} orders attributed to dominant category.

        **Active filters**: {fs.date_start} → {fs.date_end} •
        statuses: {", ".join(fs.order_statuses) if fs.order_statuses else "all"} •
        customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}
        """
    )


# ### `05_payments(1).py` — Payments page

# In[ ]:


"""
pages/05_payments.py
====================
Payments page.

Data grain: order-level master only (one row per order).
No item-level data is used here.

Key rules
---------
- payment_type_share uses primary_payment_type (the dominant method per order).
- installment_distribution applies to credit-card orders only.
- aov_by_installment_band uses credit-card orders only.
- total_payment_value vs total_price difference is shown as a reconciliation
  metric only — no causal attribution is made.
"""

import streamlit as st

from src.data_loader import load_master, load_item_master
from src.filters import render_sidebar_filters, apply_master_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)

# All statuses for payment method share (payment exists even for cancelled orders)
fs_all  = FilterState(
    date_start=fs.date_start, date_end=fs.date_end,
    include_2016=fs.include_2016, order_statuses=[],
    customer_states=fs.customer_states,
)
master_all = apply_master_filters(master, fs_all)

# Delivered orders for AOV comparisons
master_f = apply_master_filters(master, fs)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("💳 Payments")
st.caption(
    "All payment metrics use the order-level master (one row per order). "
    "Payment-vs-sales difference is a reconciliation metric only — no causal attribution."
)

if len(master_all) == 0:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI scorecards
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

pts   = kpis.payment_type_share(master_all)
n_cc  = int((master_all["primary_payment_type"] == "credit_card").sum())
n_all = len(master_all)

cc_row   = pts[pts["primary_payment_type"] == "credit_card"]
cc_share = float(cc_row["share_pct"].values[0]) if len(cc_row) else 0.0
cc_aov   = float(cc_row["avg_order_value"].values[0]) if len(cc_row) else 0.0

instdist = kpis.installment_distribution(master_all)
avg_inst = (
    (instdist["max_installments"] * instdist["order_count"]).sum()
    / instdist["order_count"].sum()
    if len(instdist) > 0 else 0.0
)
high_inst = float(
    instdist.loc[instdist["max_installments"] > 6, "order_count"].sum()
    / instdist["order_count"].sum() * 100
    if len(instdist) > 0 else 0.0
)

aov_f = kpis.avg_order_value(master_f)

c1, c2, c3, c4 = st.columns(4)
c1.metric(**charts.kpi_delta_card("Credit Card Share",    cc_share,  fmt="{:.1f}%"))
c2.metric(**charts.kpi_delta_card("Avg Order Value",      aov_f,     fmt="R$ {:,.2f}"))
c3.metric(**charts.kpi_delta_card("Avg Installments (CC)", avg_inst, fmt="{:.2f}"))
c4.metric(**charts.kpi_delta_card("High-Installment (>6)", high_inst, fmt="{:.1f}%"))

st.caption(
    f"Payment type share: {n_all:,} orders (all statuses). "
    f"Installment KPIs: {n_cc:,} credit-card orders."
)

# ---------------------------------------------------------------------------
# Payment type share
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Payment Method Distribution")

col_donut, col_bar = st.columns([1, 1])
with col_donut:
    fig_donut = charts.donut_chart(
        labels=pts["primary_payment_type"].tolist(),
        values=pts["order_count"].tolist(),
        title="Payment Method Share",
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_bar:
    fig_aov_type = charts.bar_vertical(
        pts, x="primary_payment_type", y="avg_order_value",
        title="Avg Order Value by Payment Type",
        x_axis_title="Payment Type", y_axis_title="Avg Order Value (R$)",
    )
    st.plotly_chart(fig_aov_type, use_container_width=True)

# Payment method detail table
with st.expander("Payment type detail table"):
    tbl = pts.copy()
    tbl["share_pct"]        = tbl["share_pct"].map("{:.1f}%".format)
    tbl["avg_order_value"]  = tbl["avg_order_value"].map("R$ {:,.2f}".format)
    tbl["median_order_value"] = tbl["median_order_value"].map("R$ {:,.2f}".format)
    tbl = tbl.rename(columns={
        "primary_payment_type": "Payment Type", "order_count": "Orders",
        "share_pct": "Share", "avg_order_value": "Avg AOV",
        "median_order_value": "Median AOV",
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Instalment distribution (credit card only)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Instalment Distribution (Credit Card Orders)")
st.caption("Only orders with payment_type = credit_card are included.")

if len(instdist) > 0:
    fig_inst = charts.bar_vertical(
        instdist, x="max_installments", y="order_count",
        title="Instalment Count Distribution (Credit Card)",
        x_axis_title="Max Instalments", y_axis_title="Orders",
    )
    st.plotly_chart(fig_inst, use_container_width=True)
else:
    st.info("No credit-card orders in the selected period.")

# ---------------------------------------------------------------------------
# AOV by instalment band
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Average Order Value by Instalment Band (Credit Card)")
st.caption("Bands: 1 / 2–3 / 4–6 / 7–12 / 13+")

aib = kpis.aov_by_installment_band(master_all)
if len(aib) > 0:
    fig_aib = charts.bar_vertical(
        aib, x="band_label", y="avg_order_value",
        title="AOV by Instalment Band",
        x_axis_title="Instalment Band", y_axis_title="Avg Order Value (R$)",
    )
    st.plotly_chart(fig_aib, use_container_width=True)

# ---------------------------------------------------------------------------
# Payment-vs-sales reconciliation
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Payment vs Sales Reconciliation")
st.caption(
    "⚠️ The difference between `total_payment_value` and `total_price` is shown "
    "as a reconciliation metric only. No causal attribution is made."
)

pr = kpis.payment_reconciliation(master_all)
if len(pr) > 0:
    fig_recon = charts.grouped_bar(
        pr,
        x="primary_payment_type",
        y_cols=["avg_total_price", "avg_total_payment_value"],
        title="Avg Sales Value vs Payment Value by Payment Type",
        y_labels={"avg_total_price": "Avg Sales Value (R$)",
                   "avg_total_payment_value": "Avg Payment Value (R$)"},
        y_axis_title="R$",
        barmode="group",
    )
    st.plotly_chart(fig_recon, use_container_width=True)

    tbl_recon = pr.copy()
    tbl_recon["avg_total_price"]         = tbl_recon["avg_total_price"].map("R$ {:,.2f}".format)
    tbl_recon["avg_total_payment_value"] = tbl_recon["avg_total_payment_value"].map("R$ {:,.2f}".format)
    tbl_recon["avg_difference"]          = tbl_recon["avg_difference"].map("R$ {:+,.2f}".format)
    tbl_recon = tbl_recon.rename(columns={
        "primary_payment_type": "Payment Type", "order_count": "Orders",
        "avg_total_price": "Avg Sales",
        "avg_total_payment_value": "Avg Payment",
        "avg_difference": "Avg Difference",
    })
    st.dataframe(tbl_recon, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    st.markdown(
        f"""
        **Payment type share** uses all order statuses ({n_all:,} orders) in the period.
        `primary_payment_type` = the dominant payment method for each order.

        **Instalment KPIs** apply to credit-card orders only ({n_cc:,} orders).

        **Reconciliation**: `total_payment_value` may differ from `total_price` for various
        reasons not captured in this dataset. The difference is shown without attribution.

        **Active filters**: {fs.date_start} → {fs.date_end} •
        customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}
        """
    )


# ### `06_regional(1).py` — Regional analysis page

# In[ ]:


"""
pages/06_regional.py
====================
Regional Analysis page.

Data grain
----------
- Customer-state metrics (orders, sales, delivery, reviews) use the
  order-level master (one row per order).
- Seller-state metrics (items, sales) use the item-level master.
- The two DataFrames are NEVER merged to avoid double-counting.

Layout
------
1. KPI scorecards — top states by orders / sales / delivery
2. Customer state: orders & sales
3. Customer state: delivery performance
4. Customer state: review scores
5. Seller state: items & sales
6. Data notes
"""

import streamlit as st

from src.data_loader import load_master, load_item_master
from src.filters import render_sidebar_filters, apply_master_filters, apply_item_filters, FilterState
import src.kpis as kpis
import src.charts as charts

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items  = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)

master_f = apply_master_filters(master, fs)
items_f  = apply_item_filters(items, fs)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🗺️ Regional Analysis")
st.caption(
    "Customer-state metrics use the order-level master (one row per order). "
    "Seller-state metrics use the item-level master. The two grains are never merged."
)

if len(master_f) == 0:
    st.warning("No orders match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Pre-compute aggregates
# ---------------------------------------------------------------------------
state_sales   = kpis.sales_by_state(master_f)
state_del     = kpis.delivery_by_state(master_f)
state_reviews = kpis.review_by_state(master_f)
seller_state  = kpis.sales_by_seller_state(items_f)

# ---------------------------------------------------------------------------
# KPI scorecards
# ---------------------------------------------------------------------------
st.subheader("Key Performance Indicators")

top_state_orders = state_sales.iloc[0]["customer_state"] if len(state_sales) > 0 else "N/A"
top_state_sales  = state_sales.iloc[0]["customer_state"] if len(state_sales) > 0 else "N/A"
n_states         = int(state_sales["customer_state"].nunique())
top_seller_state = seller_state.iloc[0]["seller_state"] if len(seller_state) > 0 else "N/A"
top_seller_state_pct = (
    seller_state.iloc[0]["total_sales_value"] / seller_state["total_sales_value"].sum() * 100
    if len(seller_state) > 0 else 0.0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(**charts.kpi_delta_card("Active States",          n_states,                fmt="{:,}"))
c2.metric(**charts.kpi_delta_card("Top State (Orders)",     top_state_orders,        fmt="{}"))
c3.metric(**charts.kpi_delta_card("Top Seller State",       top_seller_state,        fmt="{}"))
c4.metric(**charts.kpi_delta_card("Top Seller State Share", top_seller_state_pct,    fmt="{:.1f}%"))

# ---------------------------------------------------------------------------
# Customer state — orders & sales
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Orders & Sales by Customer State")

if len(state_sales) > 0:
    col_orders, col_sales = st.columns(2)

    with col_orders:
        fig_orders = charts.bar_horizontal(
            state_sales.head(15),
            x="order_count",
            y="customer_state",
            title="Orders by Customer State (Top 15)",
            x_axis_title="Orders",
        )
        st.plotly_chart(fig_orders, use_container_width=True)

    with col_sales:
        fig_sales = charts.bar_horizontal(
            state_sales.head(15),
            x="total_sales",
            y="customer_state",
            title="Sales Value by Customer State (Top 15)",
            x_axis_title="Sales (R$)",
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    with st.expander("Customer state — orders & sales table"):
        tbl = state_sales.copy()
        tbl["total_sales"]      = tbl["total_sales"].map("R$ {:,.0f}".format)
        tbl["avg_order_value"]  = tbl["avg_order_value"].map("R$ {:,.2f}".format)
        tbl["total_freight"]    = tbl["total_freight"].map("R$ {:,.0f}".format)
        tbl = tbl.rename(columns={
            "customer_state":    "State",
            "order_count":       "Orders",
            "total_sales":       "Total Sales",
            "avg_order_value":   "Avg Order Value",
            "total_freight":     "Total Freight",
            "unique_customers":  "Unique Customers",
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Customer state — delivery performance
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Delivery Performance by Customer State")
st.caption("Denominator: orders with known delivery time (null delivery_time_days excluded).")

if len(state_del) > 0:
    col_del, col_delay = st.columns(2)

    with col_del:
        fig_del = charts.bar_horizontal(
            state_del.head(15),
            x="avg_delivery_days",
            y="customer_state",
            title="Avg Delivery Days by State (Slowest 15)",
            x_axis_title="Avg Delivery Days",
        )
        st.plotly_chart(fig_del, use_container_width=True)

    with col_delay:
        state_del_sorted = state_del.sort_values("delay_rate", ascending=False)
        fig_delay = charts.bar_horizontal(
            state_del_sorted.head(15),
            x="delay_rate",
            y="customer_state",
            title="Delay Rate by State (Highest 15)",
            x_axis_title="Delay Rate",
        )
        st.plotly_chart(fig_delay, use_container_width=True)

    with st.expander("Customer state — delivery table"):
        tbl_del = state_del.copy()
        tbl_del["avg_delivery_days"]    = tbl_del["avg_delivery_days"].map("{:.1f}".format)
        tbl_del["median_delivery_days"] = tbl_del["median_delivery_days"].map("{:.1f}".format)
        tbl_del["delay_rate"]           = tbl_del["delay_rate"].map("{:.1%}".format)
        tbl_del = tbl_del.rename(columns={
            "customer_state":        "State",
            "order_count":           "Orders",
            "avg_delivery_days":     "Avg Delivery Days",
            "median_delivery_days":  "Median Delivery Days",
            "delay_rate":            "Delay Rate",
            "n_delayed":             "Delayed",
            "n_on_time":             "On Time",
        })
        st.dataframe(tbl_del, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Customer state — review scores
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Scores by Customer State")
st.caption("Sorted by average review score ascending (worst-rated states first).")

if len(state_reviews) > 0:
    col_rev, col_low = st.columns(2)

    with col_rev:
        fig_rev = charts.bar_horizontal(
            state_reviews.head(15),
            x="avg_review_score",
            y="customer_state",
            title="Avg Review Score by State (Lowest 15)",
            x_axis_title="Avg Score (1–5)",
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_low:
        state_rev_low = state_reviews.sort_values("low_score_rate", ascending=False)
        fig_low = charts.bar_horizontal(
            state_rev_low.head(15),
            x="low_score_rate",
            y="customer_state",
            title="1–2 Star Rate by State (Highest 15)",
            x_axis_title="1–2 Star Rate",
        )
        st.plotly_chart(fig_low, use_container_width=True)

    with st.expander("Customer state — review table"):
        tbl_rev = state_reviews.copy()
        tbl_rev["avg_review_score"] = tbl_rev["avg_review_score"].map("{:.3f}".format)
        tbl_rev["low_score_rate"]   = tbl_rev["low_score_rate"].map("{:.1%}".format)
        tbl_rev["five_star_rate"]   = tbl_rev["five_star_rate"].map("{:.1%}".format)
        tbl_rev = tbl_rev.rename(columns={
            "customer_state":    "State",
            "order_count":       "Orders",
            "avg_review_score":  "Avg Score",
            "n_reviews":         "Reviews",
            "low_score_rate":    "1–2 Star Rate",
            "five_star_rate":    "5-Star Rate",
        })
        st.dataframe(tbl_rev, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Seller state — items & sales
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Items & Sales by Seller State")
st.caption(
    "Uses item-level master (item grain). "
    "Revenue = sum of item prices. SP dominates due to marketplace concentration."
)

if len(seller_state) > 0:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig_ss_items = charts.bar_horizontal(
            seller_state.head(15),
            x="item_count",
            y="seller_state",
            title="Items Sold by Seller State (Top 15)",
            x_axis_title="Items",
        )
        st.plotly_chart(fig_ss_items, use_container_width=True)

    with col_s2:
        fig_ss_sales = charts.bar_horizontal(
            seller_state.head(15),
            x="total_sales_value",
            y="seller_state",
            title="Sales Value by Seller State (Top 15)",
            x_axis_title="Sales (R$)",
        )
        st.plotly_chart(fig_ss_sales, use_container_width=True)

    with st.expander("Seller state table"):
        tbl_ss = seller_state.copy()
        tbl_ss["total_sales_value"]    = tbl_ss["total_sales_value"].map("R$ {:,.0f}".format)
        tbl_ss["avg_item_price"]       = tbl_ss["avg_item_price"].map("R$ {:,.2f}".format)
        tbl_ss["avg_freight_per_item"] = tbl_ss["avg_freight_per_item"].map("R$ {:,.2f}".format)
        tbl_ss = tbl_ss.rename(columns={
            "seller_state":         "Seller State",
            "item_count":           "Items Sold",
            "unique_sellers":       "Unique Sellers",
            "unique_orders":        "Unique Orders",
            "total_sales_value":    "Total Sales",
            "avg_item_price":       "Avg Item Price",
            "avg_freight_per_item": "Avg Freight/Item",
        })
        st.dataframe(tbl_ss, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Data notes", expanded=False):
    n_orders = len(master_f)
    n_items  = len(items_f)
    st.markdown(
        f"""
        **Customer state** metrics are computed from the order-level master
        ({n_orders:,} orders in the selected period).

        **Seller state** metrics are computed from the item-level master
        ({n_items:,} items in the selected period).
        The two grains are **never merged** to avoid double-counting.

        **Delivery** denominator: orders with non-null `delivery_time_days`.
        **Review** denominator: orders with non-null `review_score`.

        **Active filters**: {fs.date_start} → {fs.date_end} •
        customer states: {", ".join(fs.customer_states) if fs.customer_states else "all"}
        """
    )


# ### `07_sellers(1).py` — Seller intelligence page

# In[ ]:


"""
pages/07_sellers.py
===================
Seller Intelligence page.

Data grain
----------
- Seller performance metrics use the item-level master.
- One row represents one order item.
- Order-level and item-level data are not merged here to avoid double-counting.

Planned analysis
----------------
1. Seller KPI scorecards
2. Top sellers by sales value
3. Top sellers by item/order activity
4. Seller delivery performance
5. Seller freight performance
6. Seller performance comparison
7. Data notes
"""

import streamlit as st

from src.data_loader import load_master, load_item_master
from src.filters import (
    render_sidebar_filters,
    apply_master_filters,
    apply_item_filters,
    FilterState,
)
import src.kpis as kpis
import src.charts as charts


# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
master = load_master()
items = load_item_master()

fs: FilterState = render_sidebar_filters(master, items)

master_f = apply_master_filters(master, fs)
items_f = apply_item_filters(items, fs)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🏪 Seller Intelligence")
st.caption(
    "Seller performance analysis using the item-level dataset. "
    "Sales, product activity, freight, and operational metrics are evaluated "
    "without permanently merging the order and item grains."
)

if len(items_f) == 0:
    st.warning("No seller records match the current filters.")
    st.stop()


# ---------------------------------------------------------------------------
# Seller performance aggregates
# ---------------------------------------------------------------------------

seller_sales = (
    items_f.groupby("seller_id", as_index=False)
    .agg(
        total_sales_value=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        items_sold=("order_item_id", "count"),
        unique_orders=("order_id", "nunique"),
    )
)

seller_sales["avg_item_price"] = (
    seller_sales["total_sales_value"] / seller_sales["items_sold"]
)

seller_sales["freight_per_item"] = (
    seller_sales["total_freight_value"] / seller_sales["items_sold"]
)


# ---------------------------------------------------------------------------
# Seller KPI scorecards
# ---------------------------------------------------------------------------

st.subheader("Key Performance Indicators")

n_sellers = int(seller_sales["seller_id"].nunique())

total_sales = float(seller_sales["total_sales_value"].sum())

total_items = int(seller_sales["items_sold"].sum())

avg_sales_per_seller = (
    total_sales / n_sellers
    if n_sellers > 0
    else 0.0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Active Sellers",
    f"{n_sellers:,}",
)

c2.metric(
    "Seller Sales Value",
    f"R$ {total_sales:,.0f}",
)

c3.metric(
    "Items Sold",
    f"{total_items:,}",
)

c4.metric(
    "Avg Sales / Seller",
    f"R$ {avg_sales_per_seller:,.0f}",
)

# ---------------------------------------------------------------------------
# Top sellers by sales value
# ---------------------------------------------------------------------------

st.subheader("Top Sellers by Sales Value")

top_sellers_sales = seller_sales.sort_values(
    "total_sales_value",
    ascending=False,
).head(15)

fig_sales = charts.bar_horizontal(
    top_sellers_sales,
    x="total_sales_value",
    y="seller_id",
    title="Top 15 Sellers by Sales Value",
    x_axis_title="Sales Value (R$)",
)
st.plotly_chart(fig_sales, use_container_width=True)

# ---------------------------------------------------------------------------
# Top sellers by item and order activity
# ---------------------------------------------------------------------------

st.subheader("Top Sellers by Activity")

top_sellers_activity = seller_sales.sort_values(
    "items_sold",
    ascending=False,
).head(15)

fig_activity = charts.bar_horizontal(
    top_sellers_activity,
    x="items_sold",
    y="seller_id",
    title="Top 15 Sellers by Items Sold",
    x_axis_title="Items Sold",
)

st.plotly_chart(fig_activity, use_container_width=True)


# ---------------------------------------------------------------------------
# Seller freight performance
# ---------------------------------------------------------------------------

st.subheader("Seller Freight Performance")

top_sellers_freight = seller_sales.sort_values(
    "freight_per_item",
    ascending=False,
).head(15)

fig_freight = charts.bar_horizontal(
    top_sellers_freight,
    x="freight_per_item",
    y="seller_id",
    title="Top 15 Sellers by Freight per Item",
    x_axis_title="Freight per Item (R$)",
)

st.plotly_chart(fig_freight, use_container_width=True)

st.info(
    f"Seller Intelligence dataset ready: "
    f"{len(items_f):,} filtered order-item records."
)


# ### `08_ml_delay_predictor(1).py` — ML delay predictor page

# In[ ]:


import streamlit as st
import pandas as pd

from src.ml_module import (
    train_baseline_models,
    predict_probability,
)


st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# Cached model
# ============================================================

@st.cache_resource
def load_model():
    results = train_baseline_models()
    return results["logistic"]


model = load_model()


# ============================================================
# Header
# ============================================================

st.title("🤖 Delivery Delay Predictor")
st.caption(
    "Leakage-safe machine learning prediction for e-commerce "
    "delivery delays."
)

st.info(
    "This model estimates the probability that an order will be "
    "delivered later than the estimated delivery date. "
    "It is an analytical prediction, not a guarantee."
)


# ============================================================
# Model performance
# ============================================================

st.subheader("Model Performance — Final Temporal Test")

c1, c2, c3, c4 = st.columns(4)

c1.metric("ROC-AUC", "0.710")
c2.metric("PR-AUC", "0.104")
c3.metric("Recall", "75.1%")
c4.metric("Precision", "9.4%")

st.caption(
    "Final test results from the untouched future-period test set. "
    "Prediction threshold = 0.50."
)


# ============================================================
# Input section
# ============================================================

st.subheader("Order Prediction Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    purchase_date = st.date_input(
        "Purchase Date",
        value=pd.Timestamp("2018-06-15").date(),
    )

    purchase_time = st.time_input(
        "Purchase Time",
        value=pd.Timestamp("14:00:00").time(),
    )

    purchase_datetime = pd.Timestamp.combine(
        purchase_date,
        purchase_time,
    )

    estimated_delivery_date = st.date_input(
        "Estimated Delivery Date",
        value=pd.Timestamp("2018-06-25").date(),
    )

    n_items = st.number_input(
        "Number of Items",
        min_value=1,
        max_value=50,
        value=1,
        step=1,
    )

    total_price = st.number_input(
        "Total Product Price (R$)",
        min_value=0.0,
        value=150.0,
        step=10.0,
    )

    total_freight = st.number_input(
        "Total Freight (R$)",
        min_value=0.0,
        value=25.0,
        step=5.0,
    )

    avg_item_price = st.number_input(
        "Average Item Price (R$)",
        min_value=0.0,
        value=150.0,
        step=10.0,
    )


with col2:
    total_payment_value = st.number_input(
        "Total Payment Value (R$)",
        min_value=0.0,
        value=175.0,
        step=10.0,
    )

    n_payment_methods = st.number_input(
        "Number of Payment Methods",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )

    max_installments = st.number_input(
        "Maximum Installments",
        min_value=1,
        max_value=12,
        value=1,
        step=1,
    )

    primary_payment_type = st.selectbox(
        "Primary Payment Type",
        [
            "credit_card",
            "boleto",
            "voucher",
            "debit_card",
            "not_defined",
        ],
    )

    product_category = st.selectbox(
        "Product Category",
        [
            "bed_bath_table",
            "health_beauty",
            "sports_leisure",
            "computers_accessories",
            "furniture_decor",
            "housewares",
            "watches_gifts",
            "telephony",
            "auto",
            "toys",
            "other",
        ],
    )


with col3:
    product_weight_g = st.number_input(
        "Product Weight (g)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
    )

    product_length_cm = st.number_input(
        "Product Length (cm)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

    product_height_cm = st.number_input(
        "Product Height (cm)",
        min_value=0.0,
        value=10.0,
        step=1.0,
    )

    product_width_cm = st.number_input(
        "Product Width (cm)",
        min_value=0.0,
        value=15.0,
        step=1.0,
    )

    states = [
        "SP", "RJ", "MG", "RS", "PR", "SC", "BA",
        "DF", "ES", "GO", "PE", "CE", "PA", "MT",
        "MA", "MS", "PB", "PI", "RN", "AL", "SE",
        "TO", "RO", "AM", "AC", "AP", "RR"
    ]

    customer_state = st.selectbox(
        "Customer State",
        states,
        index=0,
    )

    seller_state = st.selectbox(
        "Seller State",
        states,
        index=0,
    )


# ============================================================
# Feature engineering
# ============================================================

purchase_ts = pd.Timestamp(purchase_datetime)

estimated_ts = pd.Timestamp(estimated_delivery_date)

purchase_hour = purchase_ts.hour
purchase_day_of_week = purchase_ts.dayofweek
purchase_month_num = purchase_ts.month

estimated_delivery_window_days = (
    estimated_ts.normalize()
    - purchase_ts.normalize()
).days


# ============================================================
# Prediction
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Delivery Delay",
    type="primary",
    use_container_width=True,
):

    input_df = pd.DataFrame(
        [
            {
                "purchase_hour": purchase_hour,
                "purchase_day_of_week": purchase_day_of_week,
                "purchase_month_num": purchase_month_num,
                "estimated_delivery_window_days":
                    estimated_delivery_window_days,
                "n_items": n_items,
                "total_price": total_price,
                "total_freight": total_freight,
                "avg_item_price": avg_item_price,
                "total_payment_value":
                    total_payment_value,
                "n_payment_methods":
                    n_payment_methods,
                "max_installments":
                    max_installments,
                "primary_payment_type":
                    primary_payment_type,
                "product_category_name_english":
                    product_category,
                "product_weight_g":
                    product_weight_g,
                "product_length_cm":
                    product_length_cm,
                "product_height_cm":
                    product_height_cm,
                "product_width_cm":
                    product_width_cm,
                "customer_state":
                    customer_state,
                "seller_state":
                    seller_state,
                "order_purchase_timestamp":
                    purchase_ts,
            }
        ]
    )

    probability = float(
        predict_probability(
            model,
            input_df,
        )[0]
    )

    predicted_delay = probability >= 0.50

    st.subheader("Prediction Result")

    r1, r2 = st.columns(2)

    r1.metric(
        "Predicted Delay Probability",
        f"{probability:.1%}",
    )

    r2.metric(
        "Prediction",
        "Likely Delayed"
        if predicted_delay
        else "Likely On-Time",
    )

    if predicted_delay:
        st.warning(
            "The model predicts a delay probability at or above "
            "the 0.50 decision threshold."
        )
    else:
        st.success(
            "The model predicts a delay probability below "
            "the 0.50 decision threshold."
        )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    st.caption(
        "The prediction is based only on information available "
        "at the order/purchase stage. Delivery outcome fields "
        "were excluded to prevent target leakage."
    )


