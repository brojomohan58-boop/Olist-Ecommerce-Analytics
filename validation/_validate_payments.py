"""_validate_payments.py — headless validation of pages/05_payments.py."""
import sys, io, ast, time
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def passthrough_cache(*a, **k):
    if a and callable(a[0]): return a[0]
    def d(fn): return fn
    return d
st.cache_data = passthrough_cache

from src.data_loader import load_master, load_item_master
from src.filters import FilterState, apply_master_filters
import src.kpis as kpis
import src.charts as charts

passed = 0
failed = 0

def check(label, condition, detail=""):
    global passed, failed
    s = "PASS" if condition else "FAIL"
    print(f"  {s}  {label}" + (f"  [{detail}]" if detail else ""))
    if condition: passed += 1
    else:         failed += 1

def is_fig(obj): return isinstance(obj, go.Figure)

# ── 1. Syntax & wiring ────────────────────────────────────────────────────
print("=== 1. Syntax & app.py wiring ===")
for f in ["pages/05_payments.py", "app.py"]:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        check(f"Syntax OK: {f}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {f}", False, str(e))

app_src = open("app.py", encoding="utf-8").read()
check("05_payments.py in app.py",      "pages/05_payments.py" in app_src)
check("page_payments callable removed","def page_payments" not in app_src)

# ── 2. Data loading ───────────────────────────────────────────────────────
print("\n=== 2. Data loading ===")
t0 = time.time()
master = load_master()
items  = load_item_master()
print(f"  Loaded in {time.time()-t0:.1f}s")

fs       = FilterState()
fs_all   = FilterState(order_statuses=[])
master_f   = apply_master_filters(master, fs)
master_all = apply_master_filters(master, fs_all)
check("master_f non-empty",   len(master_f)   > 0, str(len(master_f)))
check("master_all non-empty", len(master_all) > 0, str(len(master_all)))

# ── 3. payment_type_share ─────────────────────────────────────────────────
print("\n=== 3. payment_type_share ===")
pts = kpis.payment_type_share(master_all)
check("returns DataFrame",             isinstance(pts, pd.DataFrame))
check("has required columns",
      all(c in pts.columns for c in
          ["primary_payment_type","order_count","share_pct",
           "avg_order_value","median_order_value"]))
check("share_pct sums to 100",         abs(pts["share_pct"].sum() - 100.0) < 0.01,
      f"{pts['share_pct'].sum():.4f}")
check("top type is credit_card",       pts.iloc[0]["primary_payment_type"] == "credit_card")
cc_row = pts[pts["primary_payment_type"] == "credit_card"]
cc_share = float(cc_row["share_pct"].values[0])
check("credit_card share ~75%",        70 < cc_share < 82, f"{cc_share:.2f}%")
# 3 'not_defined' orders have null total_price → filled to 0.0 (correct behaviour)
check("avg_order_value >= 0",          (pts["avg_order_value"] >= 0).all())

# ── 4. installment_distribution ───────────────────────────────────────────
print("\n=== 4. installment_distribution ===")
instdist = kpis.installment_distribution(master_all)
check("returns DataFrame",             isinstance(instdist, pd.DataFrame))
check("share_pct sums to 100",         abs(instdist["share_pct"].sum() - 100.0) < 0.01)
check("max_installments all 1-12",
      instdist["max_installments"].between(1, 12).all())
check("only credit-card rows counted",
      instdist["order_count"].sum() <= int((master_all["primary_payment_type"] == "credit_card").sum()),
      str(instdist["order_count"].sum()))

# weighted average instalments
n_cc = int((master_all["primary_payment_type"] == "credit_card").sum())
avg_inst = float(
    (instdist["max_installments"] * instdist["order_count"]).sum()
    / instdist["order_count"].sum()
)
check("avg instalments in [1,12]",     1 <= avg_inst <= 12, f"{avg_inst:.2f}")

# ── 5. aov_by_installment_band ────────────────────────────────────────────
print("\n=== 5. aov_by_installment_band ===")
aib = kpis.aov_by_installment_band(master_all)
check("returns DataFrame",             isinstance(aib, pd.DataFrame))
check("exactly 5 bands",              len(aib) == 5, str(len(aib)))
check("bands sorted correctly",
      aib["band_label"].tolist() == ["1", "2–3", "4–6", "7–12", "13+"])
check("all avg_order_value > 0",      (aib["avg_order_value"] > 0).all())
# Higher band → higher AOV
aov_1    = float(aib.loc[aib["band_label"] == "1",    "avg_order_value"].values[0])
aov_7_12 = float(aib.loc[aib["band_label"] == "7–12", "avg_order_value"].values[0])
check("AOV(7-12) > AOV(1)",           aov_7_12 > aov_1,
      f"band1={aov_1:.0f}  band7-12={aov_7_12:.0f}")

# ── 6. payment_reconciliation ─────────────────────────────────────────────
print("\n=== 6. payment_reconciliation ===")
pr = kpis.payment_reconciliation(master_all)
check("returns DataFrame",            isinstance(pr, pd.DataFrame))
check("has avg_difference column",    "avg_difference" in pr.columns)
check("has required columns",
      all(c in pr.columns for c in
          ["primary_payment_type","order_count",
           "avg_total_price","avg_total_payment_value","avg_difference"]))
check("avg_total_price > 0",          (pr["avg_total_price"] > 0).all())
check("avg_total_payment_value > 0",  (pr["avg_total_payment_value"] > 0).all())
# No causal label — just verify neutral column name
check("no 'surcharge' in column names",
      "surcharge" not in " ".join(pr.columns).lower())

# ── 7. KPI scalar helpers ─────────────────────────────────────────────────
print("\n=== 7. KPI scalar helpers ===")
aov_f = kpis.avg_order_value(master_f)
check("avg_order_value > 0",          aov_f > 0, f"R${aov_f:.2f}")

high_inst_pct = float(
    instdist.loc[instdist["max_installments"] > 6, "order_count"].sum()
    / instdist["order_count"].sum() * 100
) if len(instdist) > 0 else 0.0
check("high_inst_pct in [0,100]",     0 <= high_inst_pct <= 100, f"{high_inst_pct:.2f}%")

cc_aov = float(cc_row["avg_order_value"].values[0])
check("cc_aov > 0",                   cc_aov > 0, f"R${cc_aov:.2f}")

# ── 8. Chart outputs ──────────────────────────────────────────────────────
print("\n=== 8. Chart outputs ===")

fig_donut = charts.donut_chart(
    labels=pts["primary_payment_type"].tolist(),
    values=pts["order_count"].tolist(),
    title="Payment Method Share",
)
check("donut returns Figure",         is_fig(fig_donut))
check("donut has Pie trace",          isinstance(fig_donut.data[0], go.Pie))
check("donut label count matches pts", len(fig_donut.data[0].labels) == len(pts))

fig_aov_type = charts.bar_vertical(
    pts, x="primary_payment_type", y="avg_order_value",
    title="AOV by Payment Type",
    x_axis_title="Payment Type", y_axis_title="Avg Order Value (R$)",
)
check("AOV bar returns Figure",       is_fig(fig_aov_type))
check("AOV bar has traces",           len(fig_aov_type.data) > 0)

fig_inst = charts.bar_vertical(
    instdist, x="max_installments", y="order_count",
    title="Instalment Distribution",
    x_axis_title="Instalments", y_axis_title="Orders",
)
check("instalment bar returns Figure", is_fig(fig_inst))

fig_aib = charts.bar_vertical(
    aib, x="band_label", y="avg_order_value",
    title="AOV by Instalment Band",
    x_axis_title="Band", y_axis_title="R$",
)
check("instalment band bar Figure",   is_fig(fig_aib))
check("instalment band bar 5 bars",   len(fig_aib.data[0].x) == 5)

fig_recon = charts.grouped_bar(
    pr, x="primary_payment_type",
    y_cols=["avg_total_price", "avg_total_payment_value"],
    title="Payment Reconciliation",
    y_labels={"avg_total_price": "Avg Sales",
               "avg_total_payment_value": "Avg Payment"},
    barmode="group",
)
check("reconciliation bar Figure",    is_fig(fig_recon))
check("reconciliation has 2 traces",  len(fig_recon.data) == 2)

# ── 9. kpi_delta_card formatting ─────────────────────────────────────────
print("\n=== 9. kpi_delta_card formatting ===")
for label, val, fmt in [
    ("Credit Card Share",     cc_share,   "{:.1f}%"),
    ("Avg Order Value",       aov_f,      "R$ {:,.2f}"),
    ("Avg Instalments (CC)",  avg_inst,   "{:.2f}"),
    ("High-Instalment (>6)",  high_inst_pct, "{:.1f}%"),
]:
    card = charts.kpi_delta_card(label, val, fmt=fmt)
    check(f"card '{label}' non-empty", len(card["value"]) > 0, card["value"])

# ── 10. Grain guards ──────────────────────────────────────────────────────
print("\n=== 10. Data-grain guards ===")
check("master_f has no item price col",     "price"          not in master_f.columns)
check("master_f has total_payment_value",   "total_payment_value" in master_f.columns)
check("master_f has primary_payment_type",  "primary_payment_type" in master_f.columns)
check("master_f has max_installments",      "max_installments" in master_f.columns)

# ── 11. Edge case: empty filter ───────────────────────────────────────────
print("\n=== 11. Edge case — empty filter ===")
fs_empty   = FilterState(customer_states=["XX_NONE"])
m_empty    = apply_master_filters(master, fs_empty)
check("empty filter: 0 rows",              len(m_empty) == 0)

pts_e = kpis.payment_type_share(m_empty)
check("payment_type_share(empty) is DF",   isinstance(pts_e, pd.DataFrame))
check("payment_type_share(empty) 0 rows",  len(pts_e) == 0)

inst_e = kpis.installment_distribution(m_empty)
check("installment_distribution(empty) DF", isinstance(inst_e, pd.DataFrame))

aib_e = kpis.aov_by_installment_band(m_empty)
check("aov_by_installment_band(empty) DF", isinstance(aib_e, pd.DataFrame))

pr_e = kpis.payment_reconciliation(m_empty)
check("payment_reconciliation(empty) DF",  isinstance(pr_e, pd.DataFrame))

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {passed} passed,  {failed} failed")
if failed == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print("  SOME VALIDATIONS FAILED")
    sys.exit(1)
