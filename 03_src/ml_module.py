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