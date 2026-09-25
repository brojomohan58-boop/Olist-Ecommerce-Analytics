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
