from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from hm_ui import INK, RED, render_hero, section_label

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data/processed/customer_scores_final_2020-03-31.parquet"
HISTORY = ROOT / "models/forecasting/weekly_history.csv"
METRICS = ROOT / "models/forecasting/final_test_metrics.csv"


@st.cache_data(max_entries=4)
def customer_summary(path, modified):
    with duckdb.connect() as con:
        return con.execute("""
            SELECT COUNT(*) AS customers,
                   AVG(predicted_no_purchase_probability) AS average_risk,
                   SUM(estimated_future_spending) AS estimated_spending,
                   MIN(score_as_of_date) AS first_score_date,
                   MAX(score_as_of_date) AS last_score_date
            FROM read_parquet(?)
        """, [path]).df()


@st.cache_data(max_entries=8)
def read_csv(path, modified):
    return pd.read_csv(path)


render_hero()
section_label("01", "CUSTOMER SNAPSHOT")

history = None
if HISTORY.is_file():
    try:
        history = read_csv(str(HISTORY), HISTORY.stat().st_mtime_ns)
        needed = {"week_start", "product_group_name", "purchase_records"}
        if not needed.issubset(history.columns) or history.empty:
            raise ValueError("Weekly history is empty or has missing columns.")
        history["week_start"] = pd.to_datetime(history["week_start"], errors="raise")
        history["purchase_records"] = pd.to_numeric(history["purchase_records"], errors="raise")
    except Exception as error:
        st.error(f"Could not load weekly history: {error}")
        history = None

if SCORES.is_file():
    try:
        summary = customer_summary(str(SCORES), SCORES.stat().st_mtime_ns).iloc[0]
        if int(summary["customers"]) == 0:
            raise ValueError("The saved customer scores are empty.")
        if summary["first_score_date"] != summary["last_score_date"]:
            raise ValueError("Expected a single customer scoring date.")
        a, b, c, d = st.columns(4)
        a.metric("Scored customers", f"{int(summary['customers']):,}")
        b.metric("Average no-purchase risk", f"{summary['average_risk']:.1%}")
        c.metric("Estimated 90-day spending", f"{summary['estimated_spending']:,.2f}")
        d.metric("Complete weeks of sales", "—" if history is None else str(history['week_start'].nunique()))
        score_date = pd.Timestamp(summary["first_score_date"])
        st.html(f"""
          <div class="hm-snapshot">Customer scores: <strong>{score_date:%d %B %Y}</strong>
          · 90-day horizon · Spending in dataset units.<br>
          Risk means no recorded purchase in that window; it does not establish permanent churn.</div>
        """)
        st.html("""
          <div class="hm-model-note"><strong>Model context.</strong> The saved spending system
          underestimated aggregate spending by 26.53% in its later-period evaluation.
          Estimates are not guaranteed revenue.</div>
        """)
    except Exception as error:
        st.error(f"Could not load customer summary: {error}")
else:
    st.warning("The saved customer score file is unavailable.")

section_label("02", "RECORDED SALES")
with st.container(border=True):
    st.subheader("The weekly picture")
    st.caption("Purchase records across complete Monday–Sunday weeks. These are not distinct orders.")
    if history is not None:
        groups = sorted(history["product_group_name"].dropna().unique().tolist())
        control, spacer = st.columns([1, 1])
        with control:
            group = st.selectbox("Product group", ["All product groups"] + groups)
        if group == "All product groups":
            chart_data = history.groupby("week_start", as_index=False)["purchase_records"].sum()
        else:
            chart_data = history.loc[history["product_group_name"] == group,
                                     ["week_start", "purchase_records"]].copy()
        chart_data = chart_data.sort_values("week_start")
        plot, detail = st.columns([3, 1], gap="large")
        with plot:
            fig = px.line(chart_data, x="week_start", y="purchase_records",
                          labels={"week_start": "Week starting Monday", "purchase_records": "Purchase records"})
            fig.update_traces(line=dict(color=RED, width=2.2))
            fig.update_layout(height=340, hovermode="x unified", showlegend=False)
            st.plotly_chart(fig, width="stretch", theme=None)
        with detail:
            latest = chart_data.iloc[-1]
            peak = chart_data.loc[chart_data["purchase_records"].idxmax()]
            st.metric("Last complete week", f"{int(latest['purchase_records']):,}")
            st.caption(f"Week starting {latest['week_start']:%d %b %Y}")
            st.metric("Highest weekly count", f"{int(peak['purchase_records']):,}")
            st.caption(f"Week starting {peak['week_start']:%d %b %Y}")
        st.caption(
            f"Complete weekly history: {history['week_start'].min():%d %b %Y} – "
            f"{history['week_start'].max() + pd.Timedelta(days=6):%d %b %Y}. "
            "Use Sales analysis for the full transaction date range."
        )
    else:
        st.info("Add models/forecasting/weekly_history.csv to show the sales chart.")

section_label("03", "FROM INSIGHT TO ACTION")
for column, number, title, body, page, label in zip(
    st.columns(3),
    ["01 / EXPLORE", "02 / PREDICT", "03 / PLAN"],
    ["See the sales story.", "Understand the customer.", "Test an assumption."],
    ["Explore product groups, sales channels and purchasing patterns.",
     "Run saved risk and spending models for an individual customer.",
     "Compare targeting strategies under hypothetical campaign assumptions."],
    ["pages/1_Sales_Analysis.py", "pages/3_Customer_Predictions.py", "pages/7_Campaign_Simulator.py"],
    ["Open sales analysis →", "Open customer predictions →", "Open campaign simulator →"],
):
    with column:
        with st.container(border=True):
            st.caption(number)
            st.subheader(title)
            st.write(body)
            st.page_link(page, label=label)

with st.expander("Snapshot dates and evaluation context"):
    st.write("Customer scores: 31 March 2020, with a 90-day prediction horizon.")
    st.write("Recommendations: 31 January 2020, evaluated against 1–7 February 2020 purchases.")
    st.write("Forecasts: based on complete weekly history through 20 September 2020.")
    st.caption("The pages use different historical cutoffs. Campaign scenarios are hypothetical, not measured uplift.")
    if METRICS.is_file():
        try:
            st.dataframe(read_csv(str(METRICS), METRICS.stat().st_mtime_ns),
                         hide_index=True, width="stretch")
            st.caption("The seasonal model was selected on validation; the last-week reference performed better on the final test.")
        except Exception as error:
            st.warning(f"Saved evaluation results could not be read: {error}")
