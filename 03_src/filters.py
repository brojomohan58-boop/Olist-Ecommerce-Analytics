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