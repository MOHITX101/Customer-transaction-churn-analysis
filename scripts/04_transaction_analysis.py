"""Create aggregate purchase, channel, product and customer analysis tables."""

import argparse
import calendar
import csv
import json
import math
import os
import platform
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def query(con, sql, parameters=None):
    cursor = con.execute(sql, parameters or [])
    names = [item[0] for item in cursor.description]

    return [
        dict(zip(names, values))
        for values in cursor.fetchall()
    ]


def percent(value, total):
    return 100.0 * value / total if total else None


def growth(current, previous):
    return 100.0 * (current / previous - 1) if previous else None


def analyse(con, processed):
    for name in ["transactions", "customers", "articles"]:
        path = processed / f"{name}.parquet"

        con.execute(
            f"CREATE VIEW {name} AS "
            f"SELECT * FROM read_parquet({sql_string(path)})"
        )

    print(
        "Checking lookup keys and summarising customer activity ...",
        flush=True,
    )

    for table, key in [
        ("customers", "customer_id"),
        ("articles", "article_id"),
    ]:
        bad = query(con, f"""
            SELECT
                count(*) - count(DISTINCT {key}) AS bad_keys,
                count(*) FILTER (
                    WHERE nullif(trim({key}), '') IS NULL
                ) AS blanks
            FROM {table}
        """)[0]

        if bad["bad_keys"] or bad["blanks"]:
            raise ValueError(
                f"{table}: blank or duplicate lookup IDs. "
                "Recheck relationships."
            )

    # One temporary summary row per purchasing customer.
    # Repeated transaction rows are retained in purchase_records.
    con.execute("""
        CREATE TEMP TABLE customer_activity AS
        SELECT
            customer_id,
            count(*) AS purchase_records,
            sum(price) AS spending,
            min(t_dat) AS first_date,
            max(t_dat) AS last_date,
            bool_or(sales_channel_id = 1) AS used_channel_1,
            bool_or(sales_channel_id = 2) AS used_channel_2
        FROM transactions
        GROUP BY customer_id
    """)

    missing = query(con, """
        SELECT count(*) AS n
        FROM customer_activity a
        LEFT JOIN customers c
            ON a.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """)[0]["n"]

    if missing:
        raise ValueError(
            "Some purchasing customers have no profile. "
            "Recheck relationships."
        )

    summary = query(con, """
        SELECT
            count(*) AS purchase_records,
            sum(price) AS spending_dataset_units,
            avg(price) AS average_item_price,
            min(t_dat) AS first_date,
            max(t_dat) AS last_date,
            count(DISTINCT article_id) AS purchased_articles,

            count(*) FILTER (
                WHERE t_dat IS NULL
                    OR customer_id IS NULL
                    OR article_id IS NULL
                    OR price IS NULL
                    OR sales_channel_id IS NULL
                    OR NOT isfinite(price)
                    OR price < 0
                    OR sales_channel_id NOT IN (1, 2)
            ) AS invalid_rows

        FROM transactions
    """)[0]

    if not summary["purchase_records"] or summary["invalid_rows"]:
        raise ValueError(
            "Empty transactions or invalid required values. "
            "Recheck preparation."
        )

    summary["customer_profiles"] = con.execute(
        "SELECT count(*) FROM customers"
    ).fetchone()[0]

    summary["catalog_articles"] = con.execute(
        "SELECT count(*) FROM articles"
    ).fetchone()[0]

    summary["purchasing_customers"] = con.execute(
        "SELECT count(*) FROM customer_activity"
    ).fetchone()[0]

    summary["profiles_without_observed_purchases"] = (
        summary["customer_profiles"]
        - summary["purchasing_customers"]
    )

    summary["observed_catalog_purchase_coverage_pct"] = percent(
        summary["purchased_articles"],
        summary["catalog_articles"],
    )

    print(
        "Calculating daily trends and monthly buyer activity ...",
        flush=True,
    )

    daily = query(con, """
        SELECT
            t_dat AS purchase_date,
            count(*) AS purchase_records,
            sum(price) AS spending_dataset_units,
            avg(price) AS average_item_price
        FROM transactions
        GROUP BY t_dat
        ORDER BY t_dat
    """)

    monthly = []
    first = summary["first_date"]
    last = summary["last_date"]
    month = first.replace(day=1)

    # Count distinct buyers one month at a time to reduce peak memory.
    while month <= last:
        month_end = date(
            month.year,
            month.month,
            calendar.monthrange(month.year, month.month)[1],
        )

        start = max(first, month)
        end = min(last, month_end)

        stats = query(con, """
            SELECT
                count(*) AS purchase_records,
                count(DISTINCT customer_id) AS purchasing_customers,
                count(DISTINCT t_dat) AS days_with_purchase_records,
                coalesce(sum(price), 0) AS spending_dataset_units,
                avg(price) AS average_item_price
            FROM transactions
            WHERE t_dat BETWEEN ? AND ?
        """, [start, end])[0]

        new_buyers = con.execute("""
            SELECT count(*)
            FROM customer_activity
            WHERE first_date BETWEEN ? AND ?
        """, [start, end]).fetchone()[0]

        days = (end - start).days + 1

        stats.update({
            "month": month.isoformat()[:7],
            "covered_start": start,
            "covered_end": end,
            "covered_calendar_days": days,
            "full_calendar_month_within_date_span": (
                start == month and end == month_end
            ),
            "first_observed_purchasers": new_buyers,
            "returning_observed_purchasers": (
                stats["purchasing_customers"] - new_buyers
            ),
            "purchase_records_per_covered_day": (
                stats["purchase_records"] / days
            ),
            "spending_per_purchasing_customer": (
                stats["spending_dataset_units"]
                / stats["purchasing_customers"]
                if stats["purchasing_customers"]
                else None
            ),
        })

        previous = monthly[-1] if monthly else None

        comparable = (
            previous
            and previous["full_calendar_month_within_date_span"]
            and stats["full_calendar_month_within_date_span"]
        )

        stats["purchase_records_mom_pct"] = (
            growth(
                stats["purchase_records"],
                previous["purchase_records"],
            )
            if comparable
            else None
        )

        stats["spending_mom_pct"] = (
            growth(
                stats["spending_dataset_units"],
                previous["spending_dataset_units"],
            )
            if comparable
            else None
        )

        previous_year_month = (
            f"{month.year - 1:04d}-{month.month:02d}"
        )

        year_ago = next(
            (
                row for row in monthly
                if row["month"] == previous_year_month
            ),
            None,
        )

        comparable_year = (
            year_ago
            and year_ago["full_calendar_month_within_date_span"]
            and stats["full_calendar_month_within_date_span"]
        )

        stats["purchase_records_yoy_pct"] = (
            growth(
                stats["purchase_records"],
                year_ago["purchase_records"],
            )
            if comparable_year
            else None
        )

        monthly.append(stats)
        month = month_end + timedelta(days=1)

    print(
        "Comparing sales channels and product categories ...",
        flush=True,
    )

    channels = query(con, """
        SELECT
            sales_channel_id,
            count(*) AS purchase_records,
            count(DISTINCT customer_id) AS purchasing_customers,
            sum(price) AS spending_dataset_units,
            avg(price) AS average_item_price
        FROM transactions
        GROUP BY sales_channel_id
        ORDER BY sales_channel_id
    """)

    for row in channels:
        row["channel_label"] = {
            1: "STORE",
            2: "ONLINE",
        }[row["sales_channel_id"]]

        row["purchase_record_share_pct"] = percent(
            row["purchase_records"],
            summary["purchase_records"],
        )

        row["spending_share_pct"] = percent(
            row["spending_dataset_units"],
            summary["spending_dataset_units"],
        )

    monthly_channels = query(con, """
        SELECT
            strftime(t_dat, '%Y-%m') AS month,
            sales_channel_id,
            count(*) AS purchase_records,
            sum(price) AS spending_dataset_units
        FROM transactions
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)

    categories = query(con, """
        SELECT
            a.product_group_name,
            count(*) AS purchase_records,
            count(DISTINCT t.article_id) AS purchased_articles,
            sum(t.price) AS spending_dataset_units,
            avg(t.price) AS average_item_price,
            count(*) FILTER (
                WHERE a.article_id IS NULL
            ) AS unknown_article_rows

        FROM transactions t
        LEFT JOIN articles a
            ON t.article_id = a.article_id

        GROUP BY a.product_group_name
        ORDER BY spending_dataset_units DESC, a.product_group_name
    """)

    if any(row["unknown_article_rows"] for row in categories):
        raise ValueError(
            "Transactions contain unknown articles. "
            "Recheck relationships."
        )

    for row in categories:
        row["purchase_record_share_pct"] = percent(
            row["purchase_records"],
            summary["purchase_records"],
        )

        row["spending_share_pct"] = percent(
            row["spending_dataset_units"],
            summary["spending_dataset_units"],
        )

    top_articles = query(con, """
        SELECT
            t.article_id,
            a.prod_name,
            a.product_type_name,
            a.product_group_name,
            count(*) AS purchase_records,
            sum(t.price) AS spending_dataset_units

        FROM transactions t
        JOIN articles a
            ON t.article_id = a.article_id

        GROUP BY 1, 2, 3, 4
        ORDER BY purchase_records DESC, t.article_id
        LIMIT 50
    """)

    print(
        "Calculating repeat purchasing, recency "
        "and spending concentration ...",
        flush=True,
    )

    customer_summary = query(con, """
        SELECT
            count(*) AS purchasing_customers,

            count(*) FILTER (
                WHERE first_date < last_date
            ) AS customers_with_multiple_purchase_days,

            count(*) FILTER (
                WHERE used_channel_1 AND used_channel_2
            ) AS customers_using_both_channels,

            median(purchase_records) AS median_purchase_records,
            quantile_cont(
                purchase_records, 0.9
            ) AS p90_purchase_records,

            median(spending) AS median_spending_dataset_units,
            quantile_cont(
                spending, 0.9
            ) AS p90_spending_dataset_units,

            median(
                date_diff('day', last_date, ?)
            ) AS median_days_since_last_purchase

        FROM customer_activity
    """, [last])[0]

    concentration = query(con, """
        WITH ranked AS (
            SELECT
                spending,
                row_number() OVER (
                    ORDER BY spending DESC, customer_id
                ) AS position,
                count(*) OVER () AS customer_count
            FROM customer_activity
        )

        SELECT
            sum(spending) FILTER (
                WHERE position <= ceil(customer_count * 0.1)
            ) AS top_decile_spending,

            sum(spending) AS total_spending

        FROM ranked
    """)[0]

    customer_summary["repeat_purchase_day_customer_pct"] = percent(
        customer_summary["customers_with_multiple_purchase_days"],
        summary["purchasing_customers"],
    )

    customer_summary["both_channel_customer_pct"] = percent(
        customer_summary["customers_using_both_channels"],
        summary["purchasing_customers"],
    )

    customer_summary[
        "top_10pct_purchasing_customers_spending_share_pct"
    ] = percent(
        concentration["top_decile_spending"],
        concentration["total_spending"],
    )

    age_groups = query(con, """
        SELECT
            CASE
                WHEN c.age IS NULL THEN 'UNKNOWN'
                WHEN c.age < 25 THEN 'UNDER_25'
                WHEN c.age < 35 THEN '25_34'
                WHEN c.age < 45 THEN '35_44'
                WHEN c.age < 55 THEN '45_54'
                ELSE '55_PLUS'
            END AS age_group,

            count(*) AS purchasing_customers,
            sum(a.purchase_records) AS purchase_records,
            sum(a.spending) AS spending_dataset_units,
            avg(a.spending) AS spending_per_customer

        FROM customer_activity a
        JOIN customers c
            ON a.customer_id = c.customer_id

        GROUP BY 1
        ORDER BY 1
    """)

    tables = {
        "daily_trends": daily,
        "monthly_trends": monthly,
        "sales_channels": channels,
        "monthly_channels": monthly_channels,
        "category_performance": categories,
        "top_articles": top_articles,
        "customer_age_groups": age_groups,
    }

    checks = {}

    complete_tables = [
        "daily_trends",
        "monthly_trends",
        "sales_channels",
        "monthly_channels",
        "category_performance",
        "customer_age_groups",
    ]

    for name in complete_tables:
        values = tables[name]

        checks[name] = {
            "purchase_records_reconcile": (
                sum(row["purchase_records"] for row in values)
                == summary["purchase_records"]
            ),
            "spending_reconciles": math.isclose(
                sum(
                    row["spending_dataset_units"]
                    for row in values
                ),
                summary["spending_dataset_units"],
                rel_tol=1e-9,
                abs_tol=1e-8,
            ),
        }

        if not all(checks[name].values()):
            raise ValueError(
                f"Totals do not reconcile for {name}."
            )

    return {
        "summary": summary,
        "customer_summary": customer_summary,
        "tables": tables,
        "validation": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
    )

    args = parser.parse_args()
    processed = args.processed_dir.expanduser().resolve()

    missing = [
        name
        for name in ["transactions", "customers", "articles"]
        if not (processed / f"{name}.parquet").is_file()
    ]

    if missing:
        print(
            "Missing Parquet files: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    output_parent = ROOT / "reports"

    try:
        output_parent.mkdir(exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )

        target = output_parent / (
            f"transaction_analysis_{stamp}"
        )

        with tempfile.TemporaryDirectory(
            prefix=".analysis_work_",
            dir=output_parent,
        ) as work:
            stage = Path(work) / "results"
            stage.mkdir()

            with duckdb.connect(config={
                "memory_limit": "1GB",
                "threads": "2",
                "temp_directory": str(Path(work) / "spill"),
                "max_temp_directory_size": "2GB",
            }) as con:
                report = analyse(con, processed)

            report.update({
                "created_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "python_version": platform.python_version(),
                "duckdb_version": duckdb.__version__,
                "scope": (
                    "Descriptive full-history analysis. "
                    "Not training features or model evaluation."
                ),
                "notes": [
                    "Spending is in original dataset units; no currency or profit claim.",
                    "Purchase records are item-purchase rows, not order counts. Repeated rows are retained.",
                    "First observed purchasers are not necessarily first-ever customers.",
                    "Calendar coverage uses the first/last observed dates; it does not prove complete data collection.",
                    "Daily tables list dates with purchases; monthly per-day averages include zero-purchase days within the date span.",
                    "Growth comparisons exclude partial calendar months. Monthly customer counts are not additive.",
                    "Channel customer counts can overlap; a customer may use both channels.",
                    "The top customer decile uses ceil(10% of purchasing customers), with deterministic tie ordering.",
                    "Observed catalog purchase coverage is not inventory sell-through or a conversion rate.",
                    "Store=1 and online=2 follow the cited TUM dataset interpretation.",
                    "Customer age is a profile snapshot used descriptively, not assumed historically available.",
                    "No individual customer IDs are exported. Only the top 50 articles are listed.",
                ],
                "sources": [
                    "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data",
                    "https://www.mdsi.tum.de/fileadmin/w00cet/di-lab/pdf/TUM-DI-LAB___inovex_WS2022_Final_Report.pdf",
                ],
            })

            with (
                stage / "analysis_summary.json"
            ).open("x", encoding="utf-8") as handle:
                json.dump(
                    report,
                    handle,
                    indent=2,
                    default=str,
                    allow_nan=False,
                )
                handle.write("\n")

            for name, values in report["tables"].items():
                with (stage / f"{name}.csv").open(
                    "x",
                    encoding="utf-8-sig",
                    newline="",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=(
                            list(values[0]) if values else []
                        ),
                    )
                    writer.writeheader()
                    writer.writerows(values)

            os.rename(stage, target)

        print(f"\nAnalysis saved: {target}")
        print(
            "Upload analysis_summary.json "
            "from this folder for review."
        )
        print(
            "Purchase records: "
            f"{report['summary']['purchase_records']:,}"
        )
        print(
            "Purchasing customers: "
            f"{report['summary']['purchasing_customers']:,}"
        )
        print(
            "All aggregate purchase and spending totals reconciled. "
            "Source files were unchanged."
        )

        return 0

    except (OSError, ValueError, duckdb.Error) as exc:
        print(
            f"Analysis stopped: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)