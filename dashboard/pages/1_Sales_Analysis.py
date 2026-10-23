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

TRANSACTIONS_FILE = (
    PROJECT_ROOT / "data/processed/transactions.parquet"
)

ARTICLES_FILE = (
    PROJECT_ROOT / "data/processed/articles.parquet"
)

CHANNELS = {
    "All channels": 0,
    "Store": 1,
    "Online": 2,
}


# =========================================================
# DATA ACCESS
# =========================================================
@st.cache_data(max_entries=4)
def load_data_bounds(
    transactions_path,
    articles_path,
    versions,
):
    with duckdb.connect() as connection:
        bounds = connection.execute(
            """
            SELECT
                MIN(t_dat) AS first_date,
                MAX(t_dat) AS last_date
            FROM read_parquet(?)
            """,
            [transactions_path],
        ).df()

        article_check = connection.execute(
            """
            SELECT
                COUNT(*) AS article_rows,
                COUNT(DISTINCT article_id) AS unique_articles
            FROM read_parquet(?)
            """,
            [articles_path],
        ).df().iloc[0]

    if bounds.isna().any().any():
        raise ValueError("The transaction date range is unavailable.")

    if article_check["article_rows"] != article_check["unique_articles"]:
        raise ValueError(
            "Article IDs contain duplicates or missing values. "
            "Fix these before joining product details."
        )

    return (
        pd.Timestamp(bounds.iloc[0]["first_date"]).date(),
        pd.Timestamp(bounds.iloc[0]["last_date"]).date(),
    )


@st.cache_data(max_entries=12)
def load_sales_summary(
    transactions_path,
    articles_path,
    versions,
    start_date,
    end_date,
    channel_id,
):
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")

        transactions = connection.from_parquet(transactions_path)

        # Dates come from date widgets; channel is an integer
        # selected from the fixed CHANNELS mapping.
        condition = (
            f"t_dat >= DATE '{start_date.isoformat()}' "
            f"AND t_dat <= DATE '{end_date.isoformat()}'"
        )

        if channel_id != 0:
            condition += f" AND sales_channel_id = {int(channel_id)}"

        transactions.filter(condition).create_view("filtered_sales")

        connection.from_parquet(
            articles_path
        ).create_view("articles")

        summary = connection.execute(
            """
            SELECT
                COUNT(*) AS purchase_records,
                COUNT(DISTINCT customer_id) AS purchasing_customers,
                COALESCE(SUM(price), 0) AS sales_value,
                AVG(price) AS average_record_price
            FROM filtered_sales
            """
        ).df()

        daily = connection.execute(
            """
            SELECT
                CAST(t_dat AS DATE) AS date,
                COUNT(*) AS purchase_records,
                SUM(price) AS sales_value
            FROM filtered_sales
            GROUP BY 1
            ORDER BY 1
            """
        ).df()

        channels = connection.execute(
            """
            SELECT
                CASE
                    WHEN sales_channel_id = 1 THEN 'Store'
                    WHEN sales_channel_id = 2 THEN 'Online'
                    ELSE 'Unknown'
                END AS channel,
                COUNT(*) AS purchase_records,
                SUM(price) AS sales_value,
                AVG(price) AS average_record_price
            FROM filtered_sales
            GROUP BY 1
            ORDER BY sales_value DESC
            """
        ).df()

        groups = connection.execute(
            """
            SELECT
                COALESCE(a.product_group_name, 'Unknown')
                    AS product_group,
                COUNT(*) AS purchase_records,
                SUM(t.price) AS sales_value
            FROM filtered_sales t
            LEFT JOIN articles a
                ON t.article_id = a.article_id
            GROUP BY 1
            ORDER BY sales_value DESC
            """
        ).df()

    # Check that aggregations preserve the transaction totals.
    expected_count = int(summary.iloc[0]["purchase_records"])
    expected_sales = float(summary.iloc[0]["sales_value"])

    for table in [daily, channels, groups]:
        if int(table["purchase_records"].sum()) != expected_count:
            raise ValueError(
                "An aggregation changed the purchase-record total."
            )

        if not np.isclose(
            table["sales_value"].sum(),
            expected_sales,
            rtol=1e-9,
            atol=1e-6,
        ):
            raise ValueError(
                "An aggregation changed the sales-value total."
            )

    return summary, daily, channels, groups


# =========================================================
# HEADER AND FILE CHECKS
# =========================================================

st.title("Sales Analysis")

st.write(
    "Explore recorded sales, purchasing activity, "
    "sales channels, and product-group performance."
)

st.caption(
    "Sales value is the sum of the dataset's price field, "
    "expressed in dataset units. Purchase records are transaction "
    "rows; they are not distinct orders."
)

missing = [
    path.name
    for path in [TRANSACTIONS_FILE, ARTICLES_FILE]
    if not path.is_file()
]

if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.stop()

versions = (
    TRANSACTIONS_FILE.stat().st_mtime_ns,
    ARTICLES_FILE.stat().st_mtime_ns,
)

try:
    first_date, last_date = load_data_bounds(
        str(TRANSACTIONS_FILE),
        str(ARTICLES_FILE),
        versions,
    )
except Exception as error:
    st.error(f"Could not inspect source data: {error}")
    st.stop()


# =========================================================
# FILTERS
# =========================================================
with st.form("sales_filters"):
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        start_date = st.date_input(
            "Start date",
            value=first_date,
            min_value=first_date,
            max_value=last_date,
        )

    with filter_col2:
        end_date = st.date_input(
            "End date",
            value=last_date,
            min_value=first_date,
            max_value=last_date,
        )

    with filter_col3:
        channel_name = st.selectbox(
            "Sales channel",
            options=list(CHANNELS),
        )

    st.form_submit_button("Apply filters", type="primary")

if start_date is None or end_date is None:
    st.info("Choose both dates.")
    st.stop()

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

try:
    with st.spinner("Calculating sales summaries..."):
        summary, daily, channels, groups = load_sales_summary(
            str(TRANSACTIONS_FILE),
            str(ARTICLES_FILE),
            versions,
            start_date,
            end_date,
            CHANNELS[channel_name],
        )
except Exception as error:
    st.error(f"Could not calculate sales summaries: {error}")
    st.stop()

totals = summary.iloc[0]

if int(totals["purchase_records"]) == 0:
    st.info("No purchase records match these filters.")
    st.stop()

st.caption(
    f"Applied filters: {start_date:%d %b %Y} "
    f"to {end_date:%d %b %Y} · {channel_name}"
)


# =========================================================
# KPI CARDS
# =========================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Recorded sales value",
    f"{totals['sales_value']:,.2f}",
)

col2.metric(
    "Purchase records",
    f"{int(totals['purchase_records']):,}",
)

col3.metric(
    "Purchasing customers",
    f"{int(totals['purchasing_customers']):,}",
)

col4.metric(
    "Average price per record",
    f"{totals['average_record_price']:.5f}",
)

st.caption(
    "Purchasing customers counts distinct customers with at least "
    "one recorded purchase within the selected period and channel."
)


# =========================================================
# MONTHLY TRENDS
# =========================================================
daily["date"] = pd.to_datetime(daily["date"])

# Fill dates without recorded transactions inside the selection.
calendar = pd.date_range(
    start_date,
    end_date,
    freq="D",
    name="date",
)

daily = (
    daily.set_index("date")
    .reindex(calendar, fill_value=0)
)

monthly = daily[
    ["purchase_records", "sales_value"]
].resample("MS").sum()

monthly["selected_days"] = daily.resample("MS").size()
monthly = monthly.reset_index()

monthly["calendar_days"] = monthly["date"].dt.days_in_month

monthly["coverage"] = np.where(
    monthly["selected_days"] == monthly["calendar_days"],
    "Full calendar month",
    "Partial calendar month",
)

monthly["sales_value_per_selected_day"] = (
    monthly["sales_value"] / monthly["selected_days"]
)

coverage_colors = {
    "Full calendar month": '#C7002C',
    "Partial calendar month": '#A8794F',
}

st.divider()
st.subheader("Monthly sales and purchasing activity")

st.caption(
    "Orange bars cover only part of a calendar month, because of "
    "the date filters or dataset boundaries. Compare their totals "
    "with full months carefully."
)

sales_tab, records_tab = st.tabs([
    "Sales value",
    "Purchase records",
])

with sales_tab:
    sales_chart = px.bar(
        monthly,
        x="date",
        y="sales_value",
        color="coverage",
        color_discrete_map=coverage_colors,
        hover_data={
            "selected_days": True,
            "sales_value_per_selected_day": ":.2f",
        },
        labels={
            "date": "Month",
            "sales_value": "Sales value — dataset units",
            "coverage": "Period coverage",
            "selected_days": "Days included",
            "sales_value_per_selected_day": "Sales value per included day",
        },
    )

    st.plotly_chart(sales_chart, use_container_width=True)

with records_tab:
    records_chart = px.bar(
        monthly,
        x="date",
        y="purchase_records",
        color="coverage",
        color_discrete_map=coverage_colors,
        hover_data=["selected_days"],
        labels={
            "date": "Month",
            "purchase_records": "Purchase records",
            "coverage": "Period coverage",
            "selected_days": "Days included",
        },
    )

    st.plotly_chart(records_chart, use_container_width=True)


# =========================================================
# CHANNEL PERFORMANCE
# =========================================================
st.subheader("Sales channel performance")

channels["sales_share_pct"] = (
    100 * channels["sales_value"] / totals["sales_value"]
)

channels["purchase_share_pct"] = (
    100 * channels["purchase_records"] / totals["purchase_records"]
)

channel_chart = px.bar(
    channels,
    x="channel",
    y="sales_value",
    color="channel",
    color_discrete_map={
        "Store": '#343434',
        "Online": '#C7002C',
        "Unknown": '#6F6F6F',
    },
    hover_data={
        "purchase_records": ":,",
        "sales_share_pct": ":.2f",
        "average_record_price": ":.5f",
    },
    labels={
        "channel": "Sales channel",
        "sales_value": "Sales value — dataset units",
        "purchase_records": "Purchase records",
        "sales_share_pct": "Sales share (%)",
        "average_record_price": "Average price per record",
    },
)

channel_chart.update_layout(showlegend=False)
st.plotly_chart(channel_chart, use_container_width=True)

st.caption(
    "Channel shares are calculated within the applied filters. "
    "Select All channels to compare Store and Online."
)

with st.expander("View channel summary"):
    st.dataframe(
        channels.round(5),
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# PRODUCT GROUPS
# =========================================================
st.subheader("Top product groups by recorded sales value")

top_groups = groups.head(10).sort_values("sales_value")

group_chart = px.bar(
    top_groups,
    x="sales_value",
    y="product_group",
    orientation="h",
    hover_data={"purchase_records": ":,"},
    color_discrete_sequence=['#C7002C'],
    labels={
        "sales_value": "Sales value — dataset units",
        "product_group": "",
        "purchase_records": "Purchase records",
    },
)

group_chart.update_layout(
    height=450,
    yaxis={
        "categoryorder": "array",
        "categoryarray": top_groups["product_group"].tolist(),
    },
)

st.plotly_chart(group_chart, use_container_width=True)

with st.expander("View all product groups"):
    st.dataframe(
        groups.round(5),
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# DOWNLOAD
# =========================================================
export = monthly.copy()

export["filter_start_date"] = start_date.isoformat()
export["filter_end_date"] = end_date.isoformat()
export["channel_filter"] = channel_name

st.download_button(
    "Download monthly sales summary",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="monthly_sales_summary.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Historical recorded activity from the H&M dataset. "
    "Sales values do not establish profit or company-wide revenue."
)