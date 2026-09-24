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