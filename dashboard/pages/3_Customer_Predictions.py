from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

# This file is located inside dashboard/pages/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_CUTOFF = "2020-03-31"
OUTCOME_START = "2020-04-01"
OUTCOME_END = "2020-06-29"
HORIZON_DAYS = 90
CLASSIFICATION_THRESHOLD = 0.50

FILES = {
    "Churn model": (
        PROJECT_ROOT
        / "models/churn_enriched/catboost_churn_enriched.cbm"
    ),
    "Churn model definitions": (
        PROJECT_ROOT
        / "models/churn_enriched/catboost_churn_enriched_metadata.json"
    ),
    "Spending models": (
        PROJECT_ROOT
        / "models/spending/two_stage_spending_model.joblib"
    ),
    "Customer features": (
        PROJECT_ROOT
        / "data/processed/customer_churn_enriched_2020-03-31.parquet"
    ),
}

RESULT_KEY = "customer_predictions_page_result"


# =========================================================
# MODEL AND DATA LOADING
# =========================================================
@st.cache_resource
def load_prediction_models(
    churn_path,
    churn_modified,
    spending_path,
    spending_modified,
):
    """Load saved models; reload when their files change."""
    churn_model = CatBoostClassifier()
    churn_model.load_model(churn_path)

    spending_bundle = joblib.load(spending_path)

    if not isinstance(spending_bundle, dict):
        raise ValueError(
            "The spending model file must contain a dictionary."
        )

    required_keys = {
        "purchase_classifier",
        "spending_regressor",
        "feature_names",
    }

    missing_keys = required_keys - set(spending_bundle)

    if missing_keys:
        raise ValueError(
            "Missing spending model components: "
            + ", ".join(sorted(missing_keys))
        )

    if not churn_model.feature_names_:
        raise ValueError(
            "The churn model does not contain feature names."
        )

    if (
        spending_bundle.get("regression_target_transform") != "log1p"
        or spending_bundle.get("regression_inverse_transform") != "expm1"
    ):
        raise ValueError(
            "The spending model's transformation definitions do not "
            "match the expected log1p/expm1 workflow."
        )

    saved_horizon = spending_bundle.get("prediction_horizon_days")

    if saved_horizon is not None and int(saved_horizon) != HORIZON_DAYS:
        raise ValueError(
            f"The spending model horizon is {saved_horizon} days; "
            f"this page expects {HORIZON_DAYS} days."
        )

    return churn_model, spending_bundle


@st.cache_data(max_entries=4)
def load_example_customers(feature_path, modified):
    """Return 20 IDs for demonstration."""
    with duckdb.connect() as connection:
        result = connection.execute(
            """
            SELECT customer_id
            FROM read_parquet(?)
            WHERE customer_id IS NOT NULL
            ORDER BY customer_id
            LIMIT 20
            """,
            [feature_path],
        ).df()

    return result["customer_id"].tolist()


@st.cache_data(max_entries=100)
def load_customer_features(feature_path, modified, customer_id):
    """Retrieve one customer; a second row detects duplicates."""
    with duckdb.connect() as connection:
        return connection.execute(
            """
            SELECT *
            FROM read_parquet(?)
            WHERE customer_id = ?
            LIMIT 2
            """,
            [feature_path, customer_id],
        ).df()


# =========================================================
# INPUT VALIDATION AND INFERENCE
# =========================================================
def positive_class_index(model):
    """Locate the probability column for class 1."""
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == 1)

    if len(positions) != 1:
        raise ValueError(
            "Expected one class labelled 1. "
            f"Found: {classes.tolist()}"
        )

    return int(positions[0])


def prepare_inputs(customer, feature_names, feature_dtypes=None):
    """Select predictors in model order and validate their values."""
    feature_names = list(feature_names)

    if not feature_names:
        raise ValueError("The model feature list is empty.")

    if len(feature_names) != len(set(feature_names)):
        raise ValueError("The model feature list contains duplicates.")

    forbidden_fields = {
        "customer_id",
        "churn_90d",
        "future_spending",
        "future_spend_90d",
    }

    if forbidden_fields.intersection(feature_names):
        raise ValueError(
            "A customer ID or outcome label appears in the predictors."
        )

    missing_features = sorted(
        set(feature_names) - set(customer.columns)
    )

    if missing_features:
        raise ValueError(
            f"Missing customer predictors: {missing_features}"
        )

    inputs = customer.loc[:, feature_names].copy()

    if feature_dtypes is not None:
        if not isinstance(feature_dtypes, dict):
            raise ValueError(
                "Saved feature_dtypes must be a dictionary."
            )

        inputs = inputs.astype({
            name: feature_dtypes[name]
            for name in feature_names
            if name in feature_dtypes
        })

    if inputs.isna().any().any():
        raise ValueError(
            "The selected customer has missing predictor values."
        )

    if not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise ValueError(
            "The selected customer has non-finite predictor values."
        )

    return inputs


def generate_predictions(customer, churn_model, spending_bundle):
    """Run the churn model and the separate two-stage spending system."""
    churn_features = list(churn_model.feature_names_)

    # Match the final churn model's float32 scoring workflow.
    churn_inputs = prepare_inputs(
        customer,
        churn_features,
        {name: "float32" for name in churn_features},
    )

    churn_class_index = positive_class_index(churn_model)

    no_purchase_probability = float(
        churn_model.predict_proba(churn_inputs)[
            0, churn_class_index
        ]
    )

    # Spending has its own feature list and dtype definitions.
    spending_features = list(spending_bundle["feature_names"])

    spending_inputs = prepare_inputs(
        customer,
        spending_features,
        spending_bundle.get("feature_dtypes"),
    )

    purchase_classifier = spending_bundle["purchase_classifier"]
    spending_regressor = spending_bundle["spending_regressor"]

    purchase_class_index = positive_class_index(
        purchase_classifier
    )

    saved_class_index = spending_bundle.get("purchase_class_index")

    if (
        saved_class_index is not None
        and int(saved_class_index) != purchase_class_index
    ):
        raise ValueError(
            "The saved purchase-class index does not match "
            "the classifier's class labels."
        )

    purchase_probability = float(
        purchase_classifier.predict_proba(spending_inputs)[
            0, purchase_class_index
        ]
    )

    predicted_log_spending = float(
        spending_regressor.predict(spending_inputs)[0]
    )

    conditional_spending = float(
        np.expm1(predicted_log_spending)
    )

    if not np.isfinite(conditional_spending):
        raise ValueError(
            "The spending model returned a non-finite estimate."
        )

    conditional_spending = max(0.0, conditional_spending)

    estimated_spending = (
        purchase_probability * conditional_spending
    )

    values = [
        no_purchase_probability,
        purchase_probability,
        conditional_spending,
        estimated_spending,
    ]

    if not np.isfinite(values).all():
        raise ValueError(
            "The models returned non-finite predictions."
        )

    if not (
        0 <= no_purchase_probability <= 1
        and 0 <= purchase_probability <= 1
    ):
        raise ValueError(
            "The models returned an invalid probability."
        )

    return {
        "customer_id": str(customer.iloc[0]["customer_id"]),
        "risk": no_purchase_probability,
        "purchase_probability": purchase_probability,
        "conditional_spending": conditional_spending,
        "estimated_spending": estimated_spending,
        "predicted_no_purchase": int(
            no_purchase_probability >= CLASSIFICATION_THRESHOLD
        ),
        "inputs": churn_inputs,
    }


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Customer Predictions")

st.write(
    "Select a customer to estimate their probability of no purchase "
    "and their spending over the following 90 days."
)

st.info(
    "Historical prediction: customer history through 31 March 2020. "
    "Prediction window: 1 April–29 June 2020."
)


# =========================================================
# FILE CHECKS
# =========================================================
file_checks = []

for component, path in FILES.items():
    exists = path.is_file()

    file_checks.append({
        "Component": component,
        "File": path.name,
        "Status": "Available" if exists else "Missing",
        "Size (MiB)": (
            round(path.stat().st_size / (1024 ** 2), 3)
            if exists else None
        ),
    })

missing_components = [
    component
    for component, path in FILES.items()
    if not path.is_file()
]

with st.expander(
    "Model and data availability",
    expanded=bool(missing_components),
):
    st.dataframe(
        pd.DataFrame(file_checks),
        hide_index=True,
        use_container_width=True,
    )

if missing_components:
    st.error(
        "Required files are missing: "
        + ", ".join(missing_components)
    )
    st.stop()


# =========================================================
# LOAD MODELS AND EXAMPLE IDS
# =========================================================
churn_path = FILES["Churn model"]
spending_path = FILES["Spending models"]
feature_path = FILES["Customer features"]

resource_signature = (
    str(PROJECT_ROOT),
    churn_path.stat().st_mtime_ns,
    spending_path.stat().st_mtime_ns,
    feature_path.stat().st_mtime_ns,
)

# Discard an old displayed result if its source files changed.
previous_result = st.session_state.get(RESULT_KEY)

if (
    previous_result is not None
    and previous_result.get("resource_signature") != resource_signature
):
    st.session_state.pop(RESULT_KEY, None)

try:
    churn_model, spending_bundle = load_prediction_models(
        str(churn_path),
        churn_path.stat().st_mtime_ns,
        str(spending_path),
        spending_path.stat().st_mtime_ns,
    )

    example_customers = load_example_customers(
        str(feature_path),
        feature_path.stat().st_mtime_ns,
    )

except Exception as error:
    st.error(f"Could not load prediction resources: {error}")
    st.stop()

if not example_customers:
    st.error("The customer feature file contains no customer IDs.")
    st.stop()


# =========================================================
# CUSTOMER SELECTION
# =========================================================
st.subheader("Choose a customer")

st.caption(
    "The dropdown contains 20 example IDs. "
    "Use the text field to look up any customer in this snapshot."
)

with st.form("customer_prediction_form"):
    example_id = st.selectbox(
        "Example customer",
        options=example_customers,
    )

    entered_id = st.text_input(
        "Or paste a full customer ID",
        placeholder="Leave blank to use the selected example",
    )

    submitted = st.form_submit_button(
        "Generate predictions",
        type="primary",
    )

if submitted:
    selected_id = entered_id.strip() or example_id

    st.session_state.pop(RESULT_KEY, None)

    try:
        with st.spinner(
            "Reading customer features and generating predictions..."
        ):
            customer = load_customer_features(
                str(feature_path),
                feature_path.stat().st_mtime_ns,
                selected_id,
            )

            if customer.empty:
                raise ValueError(
                    "Customer ID was not found in the "
                    "31 March 2020 snapshot."
                )

            if len(customer) != 1:
                raise ValueError(
                    "Duplicate customer rows were found. "
                    "Prediction stopped."
                )

            result = generate_predictions(
                customer,
                churn_model,
                spending_bundle,
            )

            result["resource_signature"] = resource_signature

            st.session_state[RESULT_KEY] = result

    except Exception as error:
        st.error(f"Could not generate predictions: {error}")


# =========================================================
# DISPLAY RESULTS
# =========================================================
result = st.session_state.get(RESULT_KEY)

if result is None:
    st.caption(
        "Choose a customer and click Generate predictions to begin."
    )

else:
    st.divider()
    st.subheader("Prediction results")

    st.caption("Results for customer")
    st.code(result["customer_id"], language=None)

    st.caption(
        "History through 31 March 2020 · "
        "Prediction window: 1 April–29 June 2020"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "90-day no-purchase probability",
        f"{result['risk']:.1%}",
    )

    col2.metric(
        "Estimated 90-day spending",
        f"{result['estimated_spending']:.4f}",
    )

    col3.metric(
        "Estimated spending if purchasing",
        f"{result['conditional_spending']:.4f}",
    )

    st.caption(
        "Spending values are in dataset units, not rupees or dollars."
    )

    if result["predicted_no_purchase"] == 1:
        st.warning(
            "At the 50% classification threshold, the model predicts "
            "no purchase during the following 90 days."
        )
    else:
        st.success(
            "At the 50% classification threshold, the model predicts "
            "a purchase during the following 90 days."
        )

    st.write(
        "No-purchase risk describes inactivity during this window. "
        "It does not establish that the customer has permanently left."
    )

    st.subheader("Customer history at the prediction date")

    inputs = result["inputs"].iloc[0]

    history_col1, history_col2, history_col3 = st.columns(3)

    history_col1.metric(
        "Days since last purchase",
        f"{int(inputs['recency_days']):,}",
    )

    history_col2.metric(
        "Historical purchase days",
        f"{int(inputs['purchase_days']):,}",
    )

    history_col3.metric(
        "Historical spending",
        f"{float(inputs['historical_spending']):.4f}",
    )

    st.caption(
        "Purchase days count distinct dates with purchases. "
        "Historical spending is also in dataset units."
    )

    with st.expander("How the spending estimate is calculated"):
        st.write(
            "The spending system combines its purchase-probability "
            "classifier with a spending model trained on purchasers."
        )

        st.code(
            f"Purchase probability: "
            f"{result['purchase_probability']:.4f}\n"
            f"Estimated spending if purchasing: "
            f"{result['conditional_spending']:.4f}\n\n"
            f"Estimated future spending = "
            f"{result['purchase_probability']:.4f} × "
            f"{result['conditional_spending']:.4f}\n"
            f"                          = "
            f"{result['estimated_spending']:.4f}",
            language=None,
        )

        st.write(
            "The spending model's purchase probability comes from "
            "a separate classifier. It can differ from one minus "
            "the churn model's no-purchase probability."
        )

        st.caption(
            "The conditional spending estimate uses the saved "
            "expm1 transformation of the regression prediction. "
            "It is an estimate, not a guaranteed purchase amount."
        )

    with st.expander("View all churn model inputs"):
        feature_display = (
            result["inputs"]
            .iloc[0]
            .rename_axis("Feature")
            .reset_index(name="Value")
        )

        st.dataframe(
            feature_display,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "These are the prepared predictors supplied to the model. "
            "The observed future outcome is excluded."
        )

    export_result = pd.DataFrame([{
        "customer_id": result["customer_id"],
        "score_as_of_date": FEATURE_CUTOFF,
        "prediction_horizon_days": HORIZON_DAYS,
        "outcome_start": OUTCOME_START,
        "outcome_end": OUTCOME_END,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "predicted_no_purchase_probability": result["risk"],
        "predicted_no_purchase_90d": result["predicted_no_purchase"],
        "spending_model_purchase_probability": (
            result["purchase_probability"]
        ),
        "predicted_spend_if_purchase": (
            result["conditional_spending"]
        ),
        "estimated_future_spending": result["estimated_spending"],
        "spending_units": "dataset_units",
    }])

    st.download_button(
        label="Download this prediction",
        data=export_result.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"customer_prediction_{result['customer_id'][:12]}"
            f"_{FEATURE_CUTOFF}.csv"
        ),
        mime="text/csv",
    )

    st.warning(
        "In the later-period evaluation, the saved spending system "
        "underestimated total spending by 26.53%. "
        "That aggregate error does not determine this customer's error."
    )

st.divider()
st.caption(
    "Academic project using the H&M dataset. "
    "Predictions describe a historical evaluation period."
)