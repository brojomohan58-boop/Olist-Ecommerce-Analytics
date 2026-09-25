"""_validate_charts.py — headless validation of src/charts.py."""
import sys, io
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def passthrough_cache(*args, **kwargs):
    if args and callable(args[0]): return args[0]
    def decorator(fn): return fn
    return decorator
st.cache_data = passthrough_cache

import src.charts as charts

passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if condition else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if condition: passed += 1
    else:         failed += 1

def is_figure(obj) -> bool:
    return isinstance(obj, go.Figure)

def has_traces(fig: go.Figure) -> bool:
    return len(fig.data) > 0

def has_title(fig: go.Figure, title: str) -> bool:
    return fig.layout.title.text == title


# ── Synthetic data ─────────────────────────────────────────────────────────

months = [f"2017-{m:02d}" for m in range(1, 13)] + [f"2018-{m:02d}" for m in range(1, 9)]
trend_df = pd.DataFrame({
    "order_month_str": months,
    "order_count":     np.random.randint(500, 4000, len(months)),
    "total_sales":     np.random.uniform(50_000, 400_000, len(months)),
    "avg_order_value": np.random.uniform(80, 180, len(months)),
    "avg_delivery_days": np.random.uniform(8, 18, len(months)),
    "delay_rate":      np.random.uniform(0.04, 0.15, len(months)),
})

cat_df = pd.DataFrame({
    "category":        [f"cat_{i}" for i in range(20)],
    "total_sales_value": np.random.uniform(10_000, 500_000, 20),
    "item_count":      np.random.randint(100, 10000, 20),
    "delay_rate":      np.random.uniform(0, 0.25, 20),
})

state_df = pd.DataFrame({
    "customer_state":   [f"S{i}" for i in range(27)],
    "avg_delivery_days": np.random.uniform(5, 30, 27),
    "delay_rate":       np.random.uniform(0, 0.2, 27),
    "order_count":      np.random.randint(100, 40000, 27),
    "avg_review_score": np.random.uniform(3.5, 4.5, 27),
})

review_df = pd.DataFrame({
    "review_score": [1, 2, 3, 4, 5],
    "count":        [11356, 3128, 8131, 19046, 57012],
    "share_pct":    [11.5, 3.2, 8.2, 19.3, 57.8],
})

payment_df = pd.DataFrame({
    "primary_payment_type": ["credit_card", "boleto", "voucher", "debit_card"],
    "order_count":           [75270, 19784, 2856, 1527],
    "share_pct":             [75.7, 19.9, 2.9, 1.5],
    "avg_order_value":       [155.0, 120.0, 98.0, 108.0],
    "avg_total_price":       [148.0, 118.0, 96.0, 105.0],
    "avg_total_payment_value": [157.0, 119.0, 97.0, 107.0],
    "avg_difference":        [9.0, 1.0, 1.0, 2.0],
})

install_df = pd.DataFrame({
    "band_label":       ["1", "2–3", "4–6", "7–12", "13+"],
    "order_count":      [48268, 22792, 16205, 5943, 302],
    "avg_order_value":  [83.0, 110.0, 180.0, 302.0, 520.0],
})

delay_review_df = pd.DataFrame({
    "is_delayed": [0, 1],
    "avg_review_score": [4.29, 2.57],
    "low_score_rate":   [0.09, 0.53],
    "five_star_rate":   [0.62, 0.12],
    "order_count":      [88644, 7826],
})

scatter_df = pd.DataFrame({
    "avg_delivery_days": np.random.uniform(5, 35, 27),
    "avg_review_score":  np.random.uniform(3.4, 4.6, 27),
    "order_count":       np.random.randint(100, 40000, 27),
    "customer_state":    [f"S{i}" for i in range(27)],
})

heatmap_df = pd.DataFrame({
    "primary_category": ["cat_a", "cat_a", "cat_b", "cat_b"],
    "customer_state":   ["SP", "RJ", "SP", "RJ"],
    "delay_rate":       [0.08, 0.12, 0.05, 0.09],
})

box_df = pd.DataFrame({
    "customer_state":    np.repeat([f"S{i}" for i in range(10)], 500),
    "delivery_time_days": np.random.exponential(12, 5000).clip(0, 60),
})

series_delivery = pd.Series(np.random.exponential(12, 10000).clip(0, 60))
series_installs  = pd.Series(np.random.choice(range(1, 13), 5000))


# ── 1. line_trend ─────────────────────────────────────────────────────────

print("=== 1. line_trend ===")
fig = charts.line_trend(trend_df, x="order_month_str",
                        y_cols=["order_count", "total_sales"],
                        title="Monthly Orders & Sales",
                        secondary_y_col="total_sales",
                        y_axis_title="Orders", y2_axis_title="Sales (R$)")
check("returns Figure",      is_figure(fig))
check("has 2 traces",        len(fig.data) == 2)
check("title set",           has_title(fig, "Monthly Orders & Sales"))
check("secondary y axis",    fig.data[1].yaxis == "y2")

fig2 = charts.line_trend(trend_df, x="order_month_str",
                         y_cols=["avg_delivery_days"],
                         title="Delivery Trend", add_delay_note=True)
check("delay note annotation present",
      any(_DELAY := charts._DELAY_NOTE[:20] in str(a.text)
          for a in fig2.layout.annotations))

fig3 = charts.line_trend(trend_df, x="order_month_str",
                         y_cols=["order_count"],
                         title="Single series")
check("single series works", len(fig3.data) == 1)


# ── 2. bar_horizontal ─────────────────────────────────────────────────────

print("\n=== 2. bar_horizontal ===")
fig = charts.bar_horizontal(cat_df, x="total_sales_value", y="category",
                             title="Top Categories by Sales", top_n=10)
check("returns Figure",      is_figure(fig))
check("has 1 trace",         len(fig.data) == 1)
check("top_n respected",     len(fig.data[0].y) == 10)

fig2 = charts.bar_horizontal(cat_df, x="delay_rate", y="category",
                              title="Delay Rate by Category", top_n=5,
                              add_multicategory_note=True)
check("top_n=5 respected",   len(fig2.data[0].y) == 5)
check("multicategory note present",
      any(charts._MULTICATEGORY_NOTE[:20] in str(a.text)
          for a in fig2.layout.annotations))

# top_n larger than data
fig3 = charts.bar_horizontal(cat_df, x="total_sales_value", y="category",
                              title="All categories", top_n=100)
check("top_n > data rows uses all rows", len(fig3.data[0].y) == len(cat_df))


# ── 3. bar_vertical ───────────────────────────────────────────────────────

print("\n=== 3. bar_vertical ===")
fig = charts.bar_vertical(review_df, x="review_score", y="count",
                           title="Review Score Distribution",
                           y_axis_title="Orders", x_axis_title="Score")
check("returns Figure",      is_figure(fig))
check("has traces",          has_traces(fig))
check("title set",           has_title(fig, "Review Score Distribution"))

fig2 = charts.bar_vertical(install_df, x="band_label", y="order_count",
                            title="Instalment Distribution",
                            text_col="order_count")
check("text column accepted", is_figure(fig2))


# ── 4. donut_chart ────────────────────────────────────────────────────────

print("\n=== 4. donut_chart ===")
fig = charts.donut_chart(
    labels=payment_df["primary_payment_type"].tolist(),
    values=payment_df["order_count"].tolist(),
    title="Payment Method Share",
)
check("returns Figure",      is_figure(fig))
check("has 1 Pie trace",     len(fig.data) == 1 and isinstance(fig.data[0], go.Pie))
check("hole == 0.45",        fig.data[0].hole == 0.45)
check("correct n labels",    len(fig.data[0].labels) == 4)

fig2 = charts.donut_chart(["A", "B", "C"], [10, 20, 70], "Test", hole=0.0)
check("hole=0 (full pie)",   fig2.data[0].hole == 0.0)


# ── 5. histogram ──────────────────────────────────────────────────────────

print("\n=== 5. histogram ===")
fig = charts.histogram(series_delivery, title="Delivery Time Distribution",
                       xaxis_title="Days", nbins=30, clip_upper=60)
check("returns Figure",      is_figure(fig))
check("has 1 trace",         len(fig.data) == 1)
check("clip annotation present",
      any("clipped" in str(a.text).lower() for a in fig.layout.annotations))

fig2 = charts.histogram(series_installs, title="Instalment Count",
                        xaxis_title="Instalments", nbins=12)
check("no clip annotation when no clipping",
      not any("clipped" in str(a.text).lower() for a in fig2.layout.annotations))

fig3 = charts.histogram(series_delivery, title="Delay histogram",
                        add_delay_note=True)
check("delay note on histogram",
      any(charts._DELAY_NOTE[:20] in str(a.text) for a in fig3.layout.annotations))

# NaN-safe
fig4 = charts.histogram(pd.Series([1.0, np.nan, 3.0, np.nan, 5.0]),
                        title="NaN series")
check("handles NaN series",  is_figure(fig4))


# ── 6. box_plot ───────────────────────────────────────────────────────────

print("\n=== 6. box_plot ===")
fig = charts.box_plot(box_df, x="customer_state", y="delivery_time_days",
                      title="Delivery Time by State",
                      y_axis_title="Days", max_categories=10)
check("returns Figure",       is_figure(fig))
check("has traces",           has_traces(fig))
check("max_categories honoured",
      len(set(fig.data[0].x)) <= 10)

fig2 = charts.box_plot(box_df, x="customer_state", y="delivery_time_days",
                       title="Delay box", add_delay_note=True)
check("delay note on box_plot",
      any(charts._DELAY_NOTE[:20] in str(a.text) for a in fig2.layout.annotations))


# ── 7. scatter_plot ───────────────────────────────────────────────────────

print("\n=== 7. scatter_plot ===")
fig = charts.scatter_plot(scatter_df, x="avg_delivery_days",
                           y="avg_review_score", title="Delivery vs Review",
                           color="customer_state",
                           hover_name="customer_state")
check("returns Figure",      is_figure(fig))
check("has traces",          has_traces(fig))

# max_points sampling
large = pd.DataFrame({
    "x": np.random.randn(10000),
    "y": np.random.randn(10000),
})
fig2 = charts.scatter_plot(large, x="x", y="y", title="Large scatter",
                            max_points=500)
total_pts = sum(len(t.x) for t in fig2.data)
check("max_points sampling applied", total_pts <= 500)


# ── 8. heatmap ────────────────────────────────────────────────────────────

print("\n=== 8. heatmap ===")
fig = charts.heatmap(heatmap_df, x="customer_state", y="primary_category",
                      values="delay_rate", title="Delay Rate Heatmap",
                      colorscale="Blues")
check("returns Figure",      is_figure(fig))
check("has 1 Heatmap trace", len(fig.data) == 1 and isinstance(fig.data[0], go.Heatmap))
check("correct z shape",     len(fig.data[0].z) == 2)   # 2 categories × 2 states

# Single-row pivot (edge case)
single_row = pd.DataFrame({"cat": ["a"], "state": ["SP"], "val": [0.1]})
fig2 = charts.heatmap(single_row, x="state", y="cat", values="val",
                       title="Single cell")
check("single-cell heatmap works", is_figure(fig2))


# ── 9. grouped_bar ────────────────────────────────────────────────────────

print("\n=== 9. grouped_bar ===")
fig = charts.grouped_bar(
    delay_review_df,
    x="is_delayed",
    y_cols=["avg_review_score"],
    title="Review Score: On-time vs Delayed",
    y_axis_title="Avg Review Score",
)
check("returns Figure",      is_figure(fig))
check("has 1 trace",         len(fig.data) == 1)

fig2 = charts.grouped_bar(
    payment_df,
    x="primary_payment_type",
    y_cols=["avg_total_price", "avg_total_payment_value"],
    title="Payment Reconciliation",
    y_labels={"avg_total_price": "Sales Value",
               "avg_total_payment_value": "Payment Value"},
    barmode="group",
)
check("grouped mode, 2 traces", len(fig2.data) == 2)
check("legend names set",
      fig2.data[0].name == "Sales Value")

fig3 = charts.grouped_bar(payment_df, x="primary_payment_type",
                           y_cols=["order_count"],
                           title="Stacked test", barmode="stack")
check("stacked barmode set",  fig3.layout.barmode == "stack")


# ── 10. kpi_delta_card ────────────────────────────────────────────────────

print("\n=== 10. kpi_delta_card ===")
card = charts.kpi_delta_card("AOV", 137.75, fmt="R$ {:,.2f}")
check("returns dict",         isinstance(card, dict))
check("label key present",    card["label"] == "AOV")
check("value formatted",      card["value"] == "R$ 137.75")
check("delta is None when not set", card["delta"] is None)

card2 = charts.kpi_delta_card("Rate", 0.918, delta=0.012, fmt="{:.1%}",
                               delta_fmt="{:+.3f}")
check("percentage fmt works", card2["value"] == "91.8%")
check("delta formatted",      card2["delta"] == "+0.012")

card3 = charts.kpi_delta_card("Orders", 96211)
check("int value formatted",  card3["value"] == "96,211")


# ── 11. Palette and shared config ─────────────────────────────────────────

print("\n=== 11. Shared config ===")
check("_PALETTE is a list",         isinstance(charts._PALETTE, list))
check("_PALETTE has >= 8 colours",  len(charts._PALETTE) >= 8)
check("_DELAY_NOTE is non-empty",   len(charts._DELAY_NOTE) > 10)
check("_MULTICATEGORY_NOTE non-empty", len(charts._MULTICATEGORY_NOTE) > 10)

# All chart functions are importable
for fn_name in ["line_trend", "bar_horizontal", "bar_vertical", "donut_chart",
                "histogram", "box_plot", "scatter_plot", "heatmap",
                "grouped_bar", "kpi_delta_card"]:
    check(f"'{fn_name}' callable", callable(getattr(charts, fn_name, None)))


# ── Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED")
    sys.exit(1)
