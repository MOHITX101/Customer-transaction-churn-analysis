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

SCORES_FILE = (
    PROJECT_ROOT
    / "data/processed/customer_scores_final_2020-03-31.parquet"
)


# =========================================================
# LOAD POPULATION SUMMARY AND RISK DISTRIBUTION
# =========================================================
@st.cache_data(max_entries=4)
def load_population_summary(path, modified):
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")

        connection.from_parquet(path).create_view("scores")

        summary = connection.execute(
            """
            SELECT
                COUNT(*) AS customers,
                COUNT(DISTINCT customer_id) AS unique_customers,

                AVG(predicted_no_purchase_probability)
                    AS average_risk,

                SUM(estimated_future_spending)
                    AS total_estimated_spending,

                QUANTILE_CONT(predicted_spend_if_purchase, 0.75)
                    AS conditional_spending_q75,

                MIN(score_as_of_date) AS first_score_date,
                MAX(score_as_of_date) AS last_score_date,

                MIN(prediction_horizon_days) AS minimum_horizon,
                MAX(prediction_horizon_days) AS maximum_horizon,

                SUM(
                    CASE
                        WHEN customer_id IS NULL
                          OR score_as_of_date IS NULL
                          OR prediction_horizon_days IS NULL

                          OR predicted_no_purchase_probability IS NULL
                          OR NOT isfinite(
                              predicted_no_purchase_probability
                          )
                          OR predicted_no_purchase_probability < 0
                          OR predicted_no_purchase_probability > 1

                          OR predicted_spend_if_purchase IS NULL
                          OR NOT isfinite(predicted_spend_if_purchase)
                          OR predicted_spend_if_purchase < 0

                          OR estimated_future_spending IS NULL
                          OR NOT isfinite(estimated_future_spending)
                          OR estimated_future_spending < 0

                        THEN 1 ELSE 0
                    END
                ) AS invalid_rows
            FROM scores
            """
        ).df()

        row = summary.iloc[0]

        if int(row["customers"]) == 0:
            raise ValueError("The score file is empty.")

        if int(row["customers"]) != int(row["unique_customers"]):
            raise ValueError(
                "Customer IDs contain duplicates or missing values."
            )

        if int(row["invalid_rows"]) != 0:
            raise ValueError(
                "Some customer scores contain missing or invalid values."
            )

        if row["first_score_date"] != row["last_score_date"]:
            raise ValueError(
                "This page expects one scoring date per snapshot."
            )

        if (
            int(row["minimum_horizon"]) != 90
            or int(row["maximum_horizon"]) != 90
        ):
            raise ValueError(
                "This page expects a 90-day prediction horizon."
            )

        risk_distribution = connection.execute(
            """
            SELECT
                LEAST(
                    9,
                    FLOOR(predicted_no_purchase_probability * 10)
                )::INTEGER AS risk_bin,
                COUNT(*) AS customers
            FROM scores
            GROUP BY 1
            ORDER BY 1
            """
        ).df()

    return summary, risk_distribution


# =========================================================
# SEGMENT CUSTOMERS USING SAVED SCORES
# =========================================================
@st.cache_data(max_entries=20)
def load_segments(
    path,
    modified,
    risk_threshold,
    spending_threshold,
):
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")

        segments = connection.execute(
            """
            WITH classified AS (
                SELECT
                    predicted_no_purchase_probability,
                    predicted_spend_if_purchase,
                    estimated_future_spending,

                    CASE
                        WHEN predicted_no_purchase_probability >= ?
                        THEN 'Higher risk'
                        ELSE 'Lower risk'
                    END AS risk_group,

                    CASE
                        WHEN predicted_spend_if_purchase >= ?
                        THEN 'Higher spending potential'
                        ELSE 'Lower spending potential'
                    END AS spending_group

                FROM read_parquet(?)
            )

            SELECT
                risk_group,
                spending_group,
                COUNT(*) AS customers,

                AVG(predicted_no_purchase_probability)
                    AS average_risk,

                AVG(predicted_spend_if_purchase)
                    AS average_conditional_spending,

                SUM(estimated_future_spending)
                    AS total_estimated_spending

            FROM classified
            GROUP BY risk_group, spending_group
            """,
            [risk_threshold, spending_threshold, path],
        ).df()

    # Keep all four groups visible, even if a group is empty.
    group_grid = pd.MultiIndex.from_product(
        [
            ["Higher risk", "Lower risk"],
            [
                "Higher spending potential",
                "Lower spending potential",
            ],
        ],
        names=["risk_group", "spending_group"],
    ).to_frame(index=False)

    segments = group_grid.merge(
        segments,
        on=["risk_group", "spending_group"],
        how="left",
        validate="one_to_one",
    )

    segments["customers"] = (
        segments["customers"].fillna(0).astype(int)
    )

    segments["total_estimated_spending"] = (
        segments["total_estimated_spending"].fillna(0)
    )

    segments["segment"] = (
        segments["risk_group"]
        + " / "
        + segments["spending_group"]
    )

    return segments


# =========================================================
# HEADER AND FILE CHECKS
# =========================================================

st.title("Customer Analysis")

st.write(
    "Explore predicted purchasing inactivity and spending potential "
    "across the scored customer population."
)

if not SCORES_FILE.is_file():
    st.error(f"Missing customer score file: {SCORES_FILE.name}")
    st.stop()

modified = SCORES_FILE.stat().st_mtime_ns

try:
    summary, risk_distribution = load_population_summary(
        str(SCORES_FILE),
        modified,
    )
except Exception as error:
    st.error(f"Could not load customer analysis: {error}")
    st.stop()

population = summary.iloc[0]

total_customers = int(population["customers"])
total_spending = float(population["total_estimated_spending"])

spending_threshold = float(
    population["conditional_spending_q75"]
)

score_date = pd.Timestamp(population["first_score_date"])
outcome_start = score_date + pd.Timedelta(days=1)
outcome_end = score_date + pd.Timedelta(days=90)

st.info(
    f"Customer scores as of {score_date:%d %B %Y}. "
    f"Prediction window: {outcome_start:%d %B %Y} "
    f"– {outcome_end:%d %B %Y}."
)


# =========================================================
# SEGMENT CONTROLS
# =========================================================
with st.form("customer_segment_controls"):
    risk_threshold_pct = st.slider(
        "Higher-risk boundary (%)",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        help=(
            "Customers at or above this predicted no-purchase "
            "probability are placed in the higher-risk group. "
            "Changing this boundary does not change model probabilities."
        ),
    )

    st.form_submit_button("Apply boundary", type="primary")

risk_threshold = risk_threshold_pct / 100

try:
    segments = load_segments(
        str(SCORES_FILE),
        modified,
        risk_threshold,
        spending_threshold,
    )
except Exception as error:
    st.error(f"Could not calculate customer segments: {error}")
    st.stop()

if int(segments["customers"].sum()) != total_customers:
    st.error("Segment customer counts do not match the population.")
    st.stop()

if not np.isclose(
    segments["total_estimated_spending"].sum(),
    total_spending,
    rtol=1e-9,
    atol=1e-6,
):
    st.error("Segment spending totals do not match the population.")
    st.stop()

segments["customer_share_pct"] = (
    100 * segments["customers"] / total_customers
)

segments["estimated_spending_share_pct"] = (
    100 * segments["total_estimated_spending"] / total_spending
    if total_spending > 0
    else np.nan
)

higher_risk_customers = int(
    segments.loc[
        segments["risk_group"] == "Higher risk",
        "customers",
    ].sum()
)


# =========================================================
# SUMMARY METRICS
# =========================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Scored customers",
    f"{total_customers:,}",
)

col2.metric(
    "Average no-purchase probability",
    f"{population['average_risk']:.1%}",
)

col3.metric(
    f"Customers at or above {risk_threshold_pct}% risk",
    f"{higher_risk_customers:,}",
)

col4.metric(
    "Estimated 90-day spending",
    f"{total_spending:,.2f}",
)

st.caption(
    "Risk means no recorded purchase during the following 90 days. "
    "Average probability is not the observed no-purchase rate. "
    "Spending is expressed in dataset units."
)


# =========================================================
# RISK DISTRIBUTION
# =========================================================
st.divider()
st.subheader("How is predicted risk distributed?")

bins = pd.DataFrame({"risk_bin": range(10)})

risk_chart_data = bins.merge(
    risk_distribution,
    on="risk_bin",
    how="left",
)

risk_chart_data["customers"] = (
    risk_chart_data["customers"].fillna(0).astype(int)
)

risk_chart_data["risk_band"] = [
    f"{start}–<{start + 10}%" if start < 90 else "90–100%"
    for start in range(0, 100, 10)
]

risk_chart_data["customer_share_pct"] = (
    100 * risk_chart_data["customers"] / total_customers
)

risk_chart = px.bar(
    risk_chart_data,
    x="risk_band",
    y="customers",
    hover_data={"customer_share_pct": ":.2f"},
    color_discrete_sequence=['#343434'],
    labels={
        "risk_band": "Predicted 90-day no-purchase probability",
        "customers": "Customers",
        "customer_share_pct": "Share of customers (%)",
    },
)

st.plotly_chart(risk_chart, use_container_width=True)

st.caption(
    "The distribution covers all scored customers. "
    "Changing the risk boundary changes segment membership, "
    "not this distribution."
)


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================
st.subheader("Customer segments")

st.write(
    f"**Higher risk:** probability at or above "
    f"{risk_threshold_pct}%. "
    f"**Higher spending potential:** estimated spending if purchasing "
    f"at or above {spending_threshold:.5f} dataset units."
)

st.caption(
    "The spending boundary is the 75th percentile across the entire "
    "saved scoring population. Ties can make the higher-potential "
    "group slightly larger than 25%. These are descriptive groups."
)

segment_colors = {
    "Higher risk": '#C7002C',
    "Lower risk": '#343434',
}

customers_tab, spending_tab = st.tabs([
    "Customers by segment",
    "Estimated spending by segment",
])

with customers_tab:
    customers_chart = px.bar(
        segments,
        x="spending_group",
        y="customers",
        color="risk_group",
        barmode="group",
        color_discrete_map=segment_colors,
        hover_data={"customer_share_pct": ":.2f"},
        labels={
            "spending_group": "Spending potential",
            "customers": "Customers",
            "risk_group": "Risk group",
            "customer_share_pct": "Share of all customers (%)",
        },
    )

    st.plotly_chart(customers_chart, use_container_width=True)

with spending_tab:
    spending_chart = px.bar(
        segments,
        x="spending_group",
        y="total_estimated_spending",
        color="risk_group",
        barmode="group",
        color_discrete_map=segment_colors,
        hover_data={"estimated_spending_share_pct": ":.2f"},
        labels={
            "spending_group": "Spending potential",
            "total_estimated_spending": (
                "Estimated 90-day spending — dataset units"
            ),
            "risk_group": "Risk group",
            "estimated_spending_share_pct": (
                "Share of total estimated spending (%)"
            ),
        },
    )

    st.plotly_chart(spending_chart, use_container_width=True)

st.caption(
    "Spending potential uses spending conditional on a purchase. "
    "Estimated future spending also incorporates the spending "
    "system's separate purchase probability."
)


# =========================================================
# TABLE AND BUSINESS INTERPRETATION
# =========================================================
with st.expander("View segment summary"):
    display_segments = segments.copy()
    display_segments["average_risk_pct"] = (
        100 * display_segments["average_risk"]
    )

    st.dataframe(
        display_segments[[
            "risk_group",
            "spending_group",
            "customers",
            "customer_share_pct",
            "average_risk_pct",
            "average_conditional_spending",
            "total_estimated_spending",
            "estimated_spending_share_pct",
        ]].round(4),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("How could these groups support decisions?"):
    st.markdown(
        """
- **Higher risk, higher spending potential:** candidates for testing
  retention offers.
- **Higher risk, lower spending potential:** candidates for testing
  lower-cost re-engagement.
- **Lower risk, higher spending potential:** candidates for testing
  relevant product recommendations.
- **Lower risk, lower spending potential:** candidates for testing
  product discovery and engagement.

These are hypotheses for campaign experiments. The segment labels
do not show which customers will respond to an intervention.
        """
    )

st.warning(
    "The saved spending system underestimated aggregate spending "
    "by 26.53% in its later-period evaluation. "
    "Segment estimates are model outputs, not guaranteed revenue."
)


# =========================================================
# EXPORT
# =========================================================
export = segments.copy()

export["score_as_of_date"] = score_date.date().isoformat()
export["prediction_horizon_days"] = 90
export["higher_risk_threshold"] = risk_threshold
export["higher_spending_threshold"] = spending_threshold
export["spending_units"] = "dataset_units"

st.download_button(
    "Download customer segment summary",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="customer_segment_dashboard_summary.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Academic customer analytics using historical H&M data."
)