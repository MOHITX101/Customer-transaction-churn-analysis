from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "models/campaign_simulator/scenario_inputs.csv"
)


# =========================================================
# LOAD SAVED STRATEGY INPUTS
# =========================================================
@st.cache_data
def load_scenario_inputs(path, modified):
    inputs = pd.read_csv(path)

    required = [
        "strategy",
        "selected_customers",
        "summed_no_purchase_risk",
        "summed_risk_value_priority",
    ]

    missing = set(required) - set(inputs.columns)

    if missing:
        raise ValueError(
            f"Missing scenario input columns: {sorted(missing)}"
        )

    inputs = inputs[required].copy()

    if inputs.empty or inputs.isna().any().any():
        raise ValueError("Scenario inputs are empty or contain missing values.")

    if inputs["strategy"].duplicated().any():
        raise ValueError("Duplicate strategy rows were found.")

    numeric_columns = required[1:]

    for column in numeric_columns:
        inputs[column] = pd.to_numeric(
            inputs[column], errors="raise"
        )

    if not np.isfinite(
        inputs[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Scenario inputs contain non-finite values.")

    counts = inputs["selected_customers"]

    if (
        (counts <= 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("Selected customer counts must be positive integers.")

    if counts.nunique() != 1:
        raise ValueError(
            "Strategies must have the same contact count "
            "for this comparison."
        )

    if (
        (inputs["summed_no_purchase_risk"] < 0).any()
        or (
            inputs["summed_no_purchase_risk"]
            > inputs["selected_customers"] + 1e-6
        ).any()
    ):
        raise ValueError("The summed risk values are invalid.")

    if (inputs["summed_risk_value_priority"] < 0).any():
        raise ValueError("Risk-value totals cannot be negative.")

    inputs["selected_customers"] = counts.astype(int)

    return inputs


# =========================================================
# HEADER AND SETUP
# =========================================================

st.title("Campaign Simulator")

st.write(
    "Compare customer-targeting strategies using the saved "
    "90-day risk and spending estimates."
)

st.info(
    "Customer score date: 31 March 2020. "
    "These are hypothetical campaign scenarios, "
    "not measured campaign results."
)

if not INPUT_FILE.is_file():
    st.error(f"Missing scenario input file: {INPUT_FILE.name}")
    st.stop()

try:
    inputs = load_scenario_inputs(
        str(INPUT_FILE),
        INPUT_FILE.stat().st_mtime_ns,
    )
except Exception as error:
    st.error(f"Could not load campaign inputs: {error}")
    st.stop()

contact_count = int(inputs["selected_customers"].iloc[0])

st.caption(
    f"Each saved strategy selects {contact_count:,} customers. "
    "Strategies are alternatives and may include overlapping customers."
)


# =========================================================
# SCENARIO ASSUMPTIONS
# =========================================================
st.subheader("Set your assumptions")

left, right = st.columns(2)

with left:
    conversion_pct = st.slider(
        "Assumed conversion of otherwise non-buyers (%)",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        help=(
            "The assumed fraction of customers who would otherwise "
            "make no purchase, but purchase because of the campaign. "
            "The same assumption is applied to every strategy."
        ),
    )

with right:
    margin_pct = st.slider(
        "Assumed contribution margin before campaign costs (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help=(
            "The share of incremental spending remaining after "
            "non-campaign variable costs. This margin is assumed; "
            "the dataset does not provide it."
        ),
    )

st.caption(
    "For example, 5% means converting five out of every hundred "
    "otherwise non-buying customers—not 5% of everyone contacted."
)

conversion_fraction = conversion_pct / 100
margin_fraction = margin_pct / 100


# =========================================================
# CALCULATE SCENARIOS
# =========================================================
results = inputs.copy()

results["assumed_conversion_pct"] = conversion_pct
results["assumed_margin_pct"] = margin_pct

results["scenario_incremental_purchasers"] = (
    conversion_fraction
    * results["summed_no_purchase_risk"]
)

results["scenario_incremental_spending"] = (
    conversion_fraction
    * results["summed_risk_value_priority"]
)

results["break_even_total_campaign_cost"] = (
    results["scenario_incremental_spending"]
    * margin_fraction
)

results["break_even_cost_per_contact"] = (
    results["break_even_total_campaign_cost"]
    / results["selected_customers"]
)


# =========================================================
# SELECTED STRATEGY SUMMARY
# =========================================================
st.divider()
st.subheader("Explore a strategy")

strategy = st.selectbox(
    "Targeting strategy",
    options=results["strategy"].tolist(),
)

selected = results.loc[
    results["strategy"] == strategy
].iloc[0]

descriptions = {
    "Random selection": (
        "A fixed random selection provides a reference comparison."
    ),
    "Highest risk": (
        "Selects customers with the highest predicted probability "
        "of no purchase."
    ),
    "Highest expected spending": (
        "Selects customers with the highest estimated future spending."
    ),
    "Risk × conditional spending": (
        "Selects customers using no-purchase risk multiplied by "
        "estimated spending if they purchase."
    ),
}

st.write(descriptions.get(strategy, strategy))

col1, col2, col3 = st.columns(3)

col1.metric(
    "Customers contacted",
    f"{int(selected['selected_customers']):,}",
)

col2.metric(
    "Scenario extra purchasers",
    f"{selected['scenario_incremental_purchasers']:,.1f}",
)

col3.metric(
    "Scenario extra spending",
    f"{selected['scenario_incremental_spending']:,.2f}",
)

cost_col1, cost_col2 = st.columns(2)

cost_col1.metric(
    "Scenario break-even total campaign cost",
    f"{selected['break_even_total_campaign_cost']:,.4f}",
)

cost_col2.metric(
    "Scenario break-even cost per contact",
    f"{selected['break_even_cost_per_contact']:,.6f}",
)

st.caption(
    "Spending and cost figures use dataset units, not rupees or dollars. "
    "Fractional purchaser counts represent scenario estimates."
)


# =========================================================
# STRATEGY COMPARISON
# =========================================================
st.divider()
st.subheader("Compare all strategies")

chart_data = results.sort_values(
    "scenario_incremental_spending"
)

figure = px.bar(
    chart_data,
    x="scenario_incremental_spending",
    y="strategy",
    orientation="h",
    text="scenario_incremental_spending",
    labels={
        "scenario_incremental_spending": (
            "Scenario incremental spending — dataset units"
        ),
        "strategy": "",
    },
    color_discrete_sequence=['#C7002C'],
)

figure.update_traces(
    texttemplate="%{text:.2f}",
    textposition="auto",
)

figure.update_layout(
    height=350,
    yaxis={
        "categoryorder": "array",
        "categoryarray": chart_data["strategy"].tolist(),
    },
)

st.plotly_chart(figure, use_container_width=True)

display_columns = [
    "strategy",
    "selected_customers",
    "scenario_incremental_purchasers",
    "scenario_incremental_spending",
    "break_even_total_campaign_cost",
    "break_even_cost_per_contact",
]

st.dataframe(
    results[display_columns].rename(columns={
        "strategy": "Strategy",
        "selected_customers": "Contacts",
        "scenario_incremental_purchasers": "Scenario extra purchasers",
        "scenario_incremental_spending": "Scenario extra spending",
        "break_even_total_campaign_cost": "Break-even total cost",
        "break_even_cost_per_contact": "Break-even cost per contact",
    }).round(6),
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# EXPLANATION AND LIMITATIONS
# =========================================================
with st.expander("How the simulator calculates these values"):
    st.markdown(
        """
For each selected customer:

- **Scenario extra purchasers** = assumed conversion fraction ×
  predicted no-purchase probability.
- **Scenario extra spending** = assumed conversion fraction ×
  predicted no-purchase probability × estimated spending if purchasing.

The simulator adds these values across the selected customers.

**Break-even total campaign cost** = scenario extra spending ×
assumed contribution margin.

**Break-even cost per contact** = break-even total campaign cost ÷
number of contacts.
        """
    )

    st.write(
        "The break-even total is the assumed contribution available "
        "to cover all campaign costs. Any fixed campaign costs or "
        "incentives must also fit within that total."
    )

with st.expander("How to interpret the comparison"):
    st.write(
        "Risk × conditional spending directly optimizes the quantity "
        "used in this scenario calculation. Its stronger scenario "
        "result is therefore expected by construction; it is not "
        "independent evidence that the strategy causes more sales."
    )

    st.write(
        "The assumptions apply the same conversion fraction to every "
        "customer and use the spending model as a proxy for spending "
        "by converted customers. Actual campaign response may differ."
    )

    st.write(
        "A randomized campaign experiment with a control group would "
        "be needed to measure incremental purchases and spending."
    )

    st.caption(
        "The saved spending system underestimated aggregate spending "
        "by 26.53% in the later-period evaluation. That result does "
        "not establish the size or direction of error for these "
        "campaign scenarios."
    )


# =========================================================
# DOWNLOAD
# =========================================================
export = results.copy()
export.insert(0, "score_as_of_date", "2020-03-31")
export.insert(1, "prediction_horizon_days", 90)
export["value_units"] = "dataset_units"
export["result_type"] = "hypothetical_scenario"

st.download_button(
    "Download scenario comparison",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=(
        f"campaign_scenario_conversion_{conversion_pct}"
        f"_margin_{margin_pct}.csv"
    ),
    mime="text/csv",
)

st.divider()
st.caption(
    "Scenario planning only. No campaign is launched from this page."
)