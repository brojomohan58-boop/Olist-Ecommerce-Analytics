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