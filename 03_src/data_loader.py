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
