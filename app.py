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
