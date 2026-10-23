from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models/forecasting"

HISTORY_FILE = MODEL_DIR / "weekly_history.csv"
PREDICTOR_FILE = MODEL_DIR / "forecast_predictor.py"

RESULT_KEY = "weekly_forecasting_page_result"


# =========================================================
# LOAD SAVED HISTORY AND FORECASTING FUNCTION
# =========================================================
@st.cache_data
def load_history(path, modified):
    history = pd.read_csv(path)

    required = {
        "week_start",
        "product_group_name",
        "purchase_records",
    }

    missing = required - set(history.columns)

    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}")

    history["week_start"] = pd.to_datetime(
        history["week_start"], errors="raise"
    )

    history["purchase_records"] = pd.to_numeric(
        history["purchase_records"], errors="raise"
    )

    if history.empty:
        raise ValueError("The weekly history is empty.")

    if history[list(required)].isna().any().any():
        raise ValueError("The weekly history contains missing values.")

    if history.duplicated(
        ["week_start", "product_group_name"]
    ).any():
        raise ValueError("Duplicate product-group weeks were found.")

    if not np.isfinite(history["purchase_records"]).all():
        raise ValueError("Purchase counts contain non-finite values.")

    if (history["purchase_records"] < 0).any():
        raise ValueError("Purchase counts contain negative values.")

    if not (history["week_start"].dt.dayofweek == 0).all():
        raise ValueError("Week-start dates must be Mondays.")

    return history.sort_values(
        ["product_group_name", "week_start"]
    ).reset_index(drop=True)


@st.cache_resource
def load_forecast_function(path, modified):
    spec = importlib.util.spec_from_file_location(
        "retail_saved_forecast_predictor",
        path,
    )

    if spec is None or spec.loader is None:
        raise ValueError("Could not load the saved forecasting module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "seasonal_naive_forecast"):
        raise ValueError(
            "The saved module has no seasonal_naive_forecast function."
        )

    return module.seasonal_naive_forecast


# =========================================================
# HEADER AND FILE CHECKS
# =========================================================

st.title("Weekly Forecasting")

st.write(
    "Select a product group to forecast its purchase-record count "
    "for the week after the last complete week in the saved history."
)

missing_files = [
    path.name
    for path in [HISTORY_FILE, PREDICTOR_FILE]
    if not path.is_file()
]

if missing_files:
    st.error("Missing files: " + ", ".join(missing_files))
    st.stop()

signature = (
    str(PROJECT_ROOT),
    HISTORY_FILE.stat().st_mtime_ns,
    PREDICTOR_FILE.stat().st_mtime_ns,
)

try:
    history = load_history(
        str(HISTORY_FILE),
        HISTORY_FILE.stat().st_mtime_ns,
    )

    forecast_function = load_forecast_function(
        str(PREDICTOR_FILE),
        PREDICTOR_FILE.stat().st_mtime_ns,
    )

except Exception as error:
    st.error(f"Could not load forecasting resources: {error}")
    st.stop()


# Last saved week is a complete Monday–Sunday week.
last_week_start = history["week_start"].max()
forecast_origin = last_week_start + pd.Timedelta(days=6)
target_week = forecast_origin + pd.Timedelta(days=1)
target_week_end = target_week + pd.Timedelta(days=6)

st.info(
    f"Historical forecast as of {forecast_origin:%d %B %Y}. "
    f"Forecast week: {target_week:%d %B %Y} "
    f"to {target_week_end:%d %B %Y}."
)

st.caption(
    "The target counts recorded purchase rows. "
    "It does not measure distinct orders, stock requirements, "
    "or demand that could not be fulfilled."
)

previous_result = st.session_state.get(RESULT_KEY)

if (
    previous_result is not None
    and previous_result.get("signature") != signature
):
    st.session_state.pop(RESULT_KEY, None)


# =========================================================
# FORECAST CONTROLS
# =========================================================
groups = sorted(history["product_group_name"].unique().tolist())

with st.form("weekly_forecast_form"):
    selected_group = st.selectbox(
        "Product group",
        options=groups,
    )

    submitted = st.form_submit_button(
        "Generate weekly forecast",
        type="primary",
    )

if submitted:
    st.session_state.pop(RESULT_KEY, None)

    try:
        with st.spinner("Generating the weekly forecast..."):
            forecast = forecast_function(
                history=history.copy(),
                forecast_origin=forecast_origin,
                groups=[selected_group],
            )

            required_output = {"prediction", "week_start"}

            if not required_output.issubset(forecast.columns):
                raise ValueError(
                    "The saved function returned unexpected columns."
                )

            if len(forecast) != 1:
                raise ValueError(
                    "Expected one forecast for the selected group."
                )

            predicted_count = float(forecast.iloc[0]["prediction"])

            if (
                not np.isfinite(predicted_count)
                or predicted_count < 0
            ):
                raise ValueError("The forecast is invalid.")

            returned_week = pd.Timestamp(
                forecast.iloc[0]["week_start"]
            )

            if returned_week != target_week:
                raise ValueError(
                    "The returned forecast week does not match "
                    "the expected next week."
                )

            st.session_state[RESULT_KEY] = {
                "group": selected_group,
                "forecast": forecast,
                "predicted_count": predicted_count,
                "signature": signature,
            }

    except Exception as error:
        st.error(f"Could not generate the forecast: {error}")


# =========================================================
# RESULTS AND CHART
# =========================================================
result = st.session_state.get(RESULT_KEY)

if result is None:
    st.caption(
        "Choose a product group and click Generate weekly forecast."
    )

else:
    group = result["group"]

    group_history = history.loc[
        history["product_group_name"] == group
    ].sort_values("week_start")

    latest_rows = group_history.loc[
        group_history["week_start"] == last_week_start
    ]

    if len(latest_rows) != 1:
        st.error("The selected group is missing its latest complete week.")
        st.stop()

    last_count = float(latest_rows.iloc[0]["purchase_records"])
    predicted_count = result["predicted_count"]

    st.divider()
    st.subheader(group)

    col1, col2 = st.columns(2)

    col1.metric(
        "Forecast purchase records",
        f"{predicted_count:,.0f}",
    )

    col2.metric(
        "Last complete week's records",
        f"{last_count:,.0f}",
    )

    st.caption(
        f"Forecast period: {target_week:%d %b %Y} "
        f"– {target_week_end:%d %b %Y}"
    )

    recent_history = group_history.tail(16)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=recent_history["week_start"],
            y=recent_history["purchase_records"],
            mode="lines+markers",
            name="Observed records",
            line=dict(color='#C7002C', width=2),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[last_week_start, target_week],
            y=[last_count, predicted_count],
            mode="lines",
            name="Forecast transition",
            line=dict(color='#A8794F', width=2, dash="dash"),
            showlegend=False,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[target_week],
            y=[predicted_count],
            mode="markers",
            name="Next-week forecast",
            marker=dict(
                color='#A8794F',
                size=13,
                symbol="diamond",
            ),
        )
    )

    figure.update_layout(
        title="Recent history and next-week forecast",
        xaxis_title="Week starting Monday",
        yaxis_title="Purchase records",
        hovermode="x unified",
        legend_title_text="",
    )

    st.plotly_chart(figure, use_container_width=True)

    with st.expander("How this forecast works"):
        source_week = target_week - pd.Timedelta(weeks=52)

        st.write(
            "The selected seasonal-naive model uses the recorded count "
            "from 52 weeks before the target week."
        )

        st.write(
            f"For this forecast, the reference week starts "
            f"**{source_week:%d %B %Y}**."
        )

        st.caption(
            "This is the forecasting method selected using validation "
            "results. It generates a point forecast; no prediction "
            "interval has been estimated."
        )

    st.warning(
        "The selected seasonal model had 23.97% WAPE on the final "
        "test period. The last-week reference baseline performed "
        "better there, with 18.06% WAPE. These are aggregate results "
        "across product groups, not this forecast's individual error."
    )

    with st.expander("View forecast details"):
        st.dataframe(
            result["forecast"],
            hide_index=True,
            use_container_width=True,
        )

    st.download_button(
        "Download forecast",
        data=result["forecast"].to_csv(index=False).encode("utf-8"),
        file_name=f"weekly_forecast_{target_week:%Y-%m-%d}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Historical forecasting demonstration using the H&M dataset."
)