from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports/churn_explanations"

SCORES_FILE = REPORT_DIR / "sample_scores.parquet"
SHAP_FILE = REPORT_DIR / "sample_shap_contributions.parquet"


# =========================================================
# LOAD SAVED EXPLANATIONS
# =========================================================
@st.cache_data
def load_explanations(
    scores_path,
    scores_modified,
    shap_path,
    shap_modified,
):
    with duckdb.connect() as connection:
        scores = connection.execute(
            "SELECT * FROM read_parquet(?)",
            [scores_path],
        ).df()

        contributions = connection.execute(
            "SELECT * FROM read_parquet(?)",
            [shap_path],
        ).df()

    required_scores = {
        "customer_id",
        "feature_cutoff",
        "predicted_no_purchase_probability",
        "raw_model_output",
        "shap_reference_log_odds",
    }

    required_shap = {
        "customer_id",
        "feature",
        "feature_value",
        "shap_log_odds",
    }

    if not required_scores.issubset(scores.columns):
        raise ValueError(
            "Missing score columns: "
            f"{sorted(required_scores - set(scores.columns))}"
        )

    if not required_shap.issubset(contributions.columns):
        raise ValueError(
            "Missing SHAP columns: "
            f"{sorted(required_shap - set(contributions.columns))}"
        )

    if scores.empty or contributions.empty:
        raise ValueError("The saved explanation files are empty.")

    if scores["customer_id"].isna().any():
        raise ValueError("A customer ID is missing.")

    if scores["customer_id"].duplicated().any():
        raise ValueError("Duplicate customers exist in the score file.")

    if contributions[["customer_id", "feature"]].isna().any().any():
        raise ValueError("Missing customer IDs or feature names.")

    if contributions.duplicated(["customer_id", "feature"]).any():
        raise ValueError("Duplicate customer-feature explanations exist.")

    score_numbers = [
        "predicted_no_purchase_probability",
        "raw_model_output",
        "shap_reference_log_odds",
    ]

    if not np.isfinite(
        scores[score_numbers].to_numpy(dtype=float)
    ).all():
        raise ValueError("Score values are missing or non-finite.")

    if not np.isfinite(
        contributions[["feature_value", "shap_log_odds"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("SHAP values are missing or non-finite.")

    probabilities = scores["predicted_no_purchase_probability"]

    if not probabilities.between(0, 1).all():
        raise ValueError("Invalid predicted probabilities.")

    if set(scores["customer_id"]) != set(contributions["customer_id"]):
        raise ValueError("Customer IDs do not match across the two files.")

    return scores, contributions


def sigmoid(value):
    # Stable conversion from model log-odds to probability.
    return float(np.exp(-np.logaddexp(0.0, -value)))


# =========================================================
# HEADER AND FILE CHECKS
# =========================================================

st.title("Model Explanations")

st.write(
    "Explore the features that contributed to a customer's "
    "predicted probability of no purchase in the next 90 days."
)

missing = [
    path.name
    for path in [SCORES_FILE, SHAP_FILE]
    if not path.is_file()
]

if missing:
    st.error("Missing explanation files: " + ", ".join(missing))
    st.stop()

try:
    scores, contributions = load_explanations(
        str(SCORES_FILE),
        SCORES_FILE.stat().st_mtime_ns,
        str(SHAP_FILE),
        SHAP_FILE.stat().st_mtime_ns,
    )
except Exception as error:
    st.error(f"Could not load explanations: {error}")
    st.stop()

st.info(
    f"This page displays saved explanations for "
    f"{len(scores):,} sampled customers. "
    "It does not represent every customer in the dataset."
)


# =========================================================
# CUSTOMER SELECTION
# =========================================================
ranked_scores = scores.sort_values(
    ["predicted_no_purchase_probability", "customer_id"],
    ascending=[False, True],
)

risk_lookup = scores.set_index("customer_id")[
    "predicted_no_purchase_probability"
].to_dict()

selected_id = st.selectbox(
    "Choose a sampled customer — highest risk first",
    options=ranked_scores["customer_id"].tolist(),
    format_func=lambda customer_id: (
        f"{risk_lookup[customer_id]:.1%} risk | {customer_id}"
    ),
)

score = scores.loc[
    scores["customer_id"] == selected_id
].iloc[0]

customer_shap = contributions.loc[
    contributions["customer_id"] == selected_id
].copy()

cutoff = pd.Timestamp(score["feature_cutoff"])
outcome_start = cutoff + pd.Timedelta(days=1)
outcome_end = cutoff + pd.Timedelta(days=90)

st.caption(
    f"History through {cutoff:%d %B %Y} · "
    f"Prediction window: {outcome_start:%d %B %Y} "
    f"– {outcome_end:%d %B %Y}"
)


# =========================================================
# VERIFY THAT THE EXPLANATION RECONSTRUCTS THE PREDICTION
# =========================================================
reference = float(score["shap_reference_log_odds"])
raw_output = float(score["raw_model_output"])
predicted_probability = float(
    score["predicted_no_purchase_probability"]
)

reconstructed_raw = (
    reference + customer_shap["shap_log_odds"].sum()
)

reconstructed_probability = sigmoid(reconstructed_raw)

if not (
    np.isclose(reconstructed_raw, raw_output, atol=1e-6, rtol=0)
    and np.isclose(
        reconstructed_probability,
        predicted_probability,
        atol=1e-6,
        rtol=0,
    )
):
    st.error(
        "The saved contributions do not reconstruct this prediction. "
        "Check that the explanation files came from the same model run."
    )
    st.stop()


# =========================================================
# INDIVIDUAL EXPLANATION
# =========================================================
col1, col2 = st.columns(2)

col1.metric(
    "Predicted 90-day no-purchase risk",
    f"{predicted_probability:.1%}",
)

col2.metric(
    "Features explained",
    len(customer_shap),
)

st.subheader("What influenced this prediction?")

st.write(
    "**Red bars** raise the model's predicted no-purchase risk. "
    "**Teal bars** lower it, relative to the model's SHAP reference."
)

customer_shap["absolute_contribution"] = (
    customer_shap["shap_log_odds"].abs()
)

ranked_features = customer_shap.sort_values(
    ["absolute_contribution", "feature"],
    ascending=[False, True],
)

top_features = ranked_features.head(10)[
    ["feature", "shap_log_odds"]
].copy()

remaining = ranked_features.iloc[10:]

if not remaining.empty:
    top_features = pd.concat(
        [
            top_features,
            pd.DataFrame([{
                "feature": "All remaining features combined",
                "shap_log_odds": remaining["shap_log_odds"].sum(),
            }]),
        ],
        ignore_index=True,
    )

top_features["effect"] = np.where(
    top_features["shap_log_odds"] >= 0,
    "Raises predicted risk",
    "Lowers predicted risk",
)

top_features = top_features.sort_values("shap_log_odds")

figure = px.bar(
    top_features,
    x="shap_log_odds",
    y="feature",
    orientation="h",
    color="effect",
    color_discrete_map={
        "Raises predicted risk": "#DC2626",
        "Lowers predicted risk": '#4B5563',
    },
    labels={
        "shap_log_odds": "Contribution to model log-odds",
        "feature": "",
        "effect": "Effect",
    },
    title="Largest contributions for this customer",
)

figure.add_vline(x=0, line_color='#6F6F6F', line_width=1)

figure.update_layout(
    height=520,
    yaxis={
        "categoryorder": "array",
        "categoryarray": top_features["feature"].tolist(),
    },
)

st.plotly_chart(figure, use_container_width=True)

st.caption(
    "Bar values are in log-odds, not percentage points. "
    "They explain the model's calculation, not the causes "
    "of a customer's behaviour."
)

with st.expander("View feature values and contributions"):
    display_table = ranked_features[
        ["feature", "feature_value", "shap_log_odds"]
    ].copy()

    display_table["effect"] = np.where(
        display_table["shap_log_odds"] > 0,
        "Raises predicted risk",
        np.where(
            display_table["shap_log_odds"] < 0,
            "Lowers predicted risk",
            "No contribution",
        ),
    )

    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "A zero purchase-gap value may be a preprocessing placeholder "
        "for insufficient purchase history. An imputed age is not "
        "an observed age; check age_was_missing."
    )

with st.expander("Check how the contributions form the prediction"):
    st.write(
        "The SHAP reference plus every feature contribution equals "
        "the model's raw output. The sigmoid function converts "
        "that output to a probability."
    )

    st.code(
        f"Reference log-odds: {reference:.6f}\n"
        f"Sum of contributions: "
        f"{customer_shap['shap_log_odds'].sum():.6f}\n"
        f"Reconstructed raw output: {reconstructed_raw:.6f}\n"
        f"Reconstructed probability: "
        f"{reconstructed_probability:.6f}\n"
        f"Saved probability: {predicted_probability:.6f}",
        language=None,
    )

    st.success("The saved explanation reconstructs the prediction.")

    st.caption(
        "The SHAP reference is a model reference. Its sigmoid "
        "is not necessarily the average customer probability."
    )

st.download_button(
    "Download this customer's explanation",
    data=customer_shap.drop(
        columns="absolute_contribution"
    ).to_csv(index=False).encode("utf-8"),
    file_name=f"customer_explanation_{selected_id[:12]}.csv",
    mime="text/csv",
)


# =========================================================
# OVERALL IMPORTANCE WITHIN THE SAVED SAMPLE
# =========================================================
st.divider()
st.subheader("Most influential features across the saved sample")

importance = (
    contributions.assign(
        absolute_shap=contributions["shap_log_odds"].abs()
    )
    .groupby("feature", as_index=False)["absolute_shap"]
    .mean()
    .rename(columns={"absolute_shap": "mean_absolute_shap"})
    .sort_values("mean_absolute_shap", ascending=False)
    .head(10)
    .sort_values("mean_absolute_shap")
)

importance_chart = px.bar(
    importance,
    x="mean_absolute_shap",
    y="feature",
    orientation="h",
    labels={
        "mean_absolute_shap": "Mean absolute SHAP contribution",
        "feature": "",
    },
    color_discrete_sequence=['#4B5563'],
)

importance_chart.update_layout(
    height=440,
    yaxis={
        "categoryorder": "array",
        "categoryarray": importance["feature"].tolist(),
    },
)

st.plotly_chart(importance_chart, use_container_width=True)

st.caption(
    "This chart summarizes contribution magnitude across the saved "
    "sample. It does not show direction or prove causation. "
    "Correlated features can share predictive information."
)