"""Audit H&M IDs, transaction links and customer field values."""

import argparse
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]

CSV_SOURCE = """read_csv(
    ?, header=true, delim=',', quote='"', escape='"',
    all_varchar=true, strict_mode=true, ignore_errors=false
)"""


def rows(con, sql, parameters=None):
    cursor = con.execute(sql, parameters or [])
    names = [item[0] for item in cursor.description]

    return [
        dict(zip(names, row))
        for row in cursor.fetchall()
    ]


def check_ids(con, table, column):
    # These names come from our fixed table definitions, not user input.
    stats = rows(con, f"""
        SELECT
            count(*) AS row_count,

            count(*) FILTER (
                WHERE nullif(trim({column}), '') IS NULL
            ) AS blank_ids,

            count(*) FILTER (
                WHERE {column} <> trim({column})
            ) AS ids_with_outer_spaces,

            count(*) FILTER (
                WHERE nullif(trim({column}), '') IS NOT NULL
            ) AS nonblank_id_rows,

            count(DISTINCT {column}) FILTER (
                WHERE nullif(trim({column}), '') IS NOT NULL
            ) AS unique_nonblank_ids

        FROM {table}
    """)[0]

    stats["duplicate_rows_beyond_first"] = (
        stats["nonblank_id_rows"]
        - stats["unique_nonblank_ids"]
    )

    stats["duplicate_id_values"] = rows(con, f"""
        SELECT count(*) AS n
        FROM (
            SELECT {column}
            FROM {table}
            WHERE nullif(trim({column}), '') IS NOT NULL
            GROUP BY {column}
            HAVING count(*) > 1
        ) AS duplicate_groups
    """)[0]["n"]

    return stats


def audit(con, data_dir):
    print(
        "Loading customer fields and article IDs ...",
        flush=True,
    )

    con.execute(f"""
        CREATE TEMP TABLE customers AS
        SELECT
            customer_id,
            FN,
            Active,
            club_member_status,
            fashion_news_frequency,
            age
        FROM {CSV_SOURCE}
    """, [str(data_dir / "customers.csv")])

    con.execute(f"""
        CREATE TEMP TABLE articles AS
        SELECT article_id
        FROM {CSV_SOURCE}
    """, [str(data_dir / "articles.csv")])

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "duckdb_version": duckdb.__version__,
        "scope": (
            "Full-file ID and relationship checks; customer field profiles. "
            "No cleaning or churn labeling."
        ),
        "id_matching": (
            "Exact text matching. ID whitespace is flagged, "
            "not silently removed."
        ),
        "customer_ids": check_ids(
            con, "customers", "customer_id"
        ),
        "article_ids": check_ids(
            con, "articles", "article_id"
        ),
    }

    # Unique lookup sets prevent duplicate IDs multiplying transaction rows.
    # The original CSVs and their repeated transaction rows remain unchanged.
    con.execute("""
        CREATE TEMP TABLE customer_keys AS
        SELECT DISTINCT customer_id
        FROM customers
        WHERE nullif(trim(customer_id), '') IS NOT NULL
    """)

    con.execute("""
        CREATE TEMP TABLE article_keys AS
        SELECT DISTINCT article_id
        FROM articles
        WHERE nullif(trim(article_id), '') IS NOT NULL
    """)

    print(
        "Checking links for every transaction ...",
        flush=True,
    )

    report["transaction_links"] = rows(con, f"""
        SELECT
            count(*) AS transaction_rows,

            count(*) FILTER (
                WHERE nullif(trim(t.customer_id), '') IS NULL
            ) AS blank_customer_ids,

            count(*) FILTER (
                WHERE nullif(trim(t.article_id), '') IS NULL
            ) AS blank_article_ids,

            count(*) FILTER (
                WHERE t.customer_id <> trim(t.customer_id)
            ) AS customer_ids_with_outer_spaces,

            count(*) FILTER (
                WHERE t.article_id <> trim(t.article_id)
            ) AS article_ids_with_outer_spaces,

            count(*) FILTER (
                WHERE nullif(trim(t.customer_id), '') IS NOT NULL
                AND c.customer_id IS NULL
            ) AS rows_with_unknown_customer,

            count(*) FILTER (
                WHERE nullif(trim(t.article_id), '') IS NOT NULL
                AND a.article_id IS NULL
            ) AS rows_with_unknown_article,

            count(*) FILTER (
                WHERE c.customer_id IS NOT NULL
                AND a.article_id IS NOT NULL
            ) AS rows_with_both_links

        FROM {CSV_SOURCE} AS t

        LEFT JOIN customer_keys AS c
            ON t.customer_id = c.customer_id

        LEFT JOIN article_keys AS a
            ON t.article_id = a.article_id
    """, [str(data_dir / "transactions_train.csv")])[0]

    print(
        "Inspecting customer field values and ages ...",
        flush=True,
    )

    report["customer_field_distributions"] = {}
    report["distribution_limit_per_field"] = 100
    report["distribution_note"] = (
        "Profiles trim outer spaces and group blanks as null; "
        "IDs remain unchanged. At most 100 categories per field are shown."
    )

    fields = [
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
    ]

    for column in fields:
        report["customer_field_distributions"][column] = rows(
            con,
            f"""
                SELECT
                    nullif(trim({column}), '') AS value,
                    count(*) AS customer_rows,
                    count(*) OVER () AS total_categories_in_field
                FROM customers
                GROUP BY 1
                ORDER BY customer_rows DESC, value ASC NULLS LAST
                LIMIT 100
            """,
        )

    report["age_profile"] = rows(con, """
        WITH ages AS (
            SELECT
                nullif(trim(age), '') AS raw_age,
                try_cast(
                    nullif(trim(age), '') AS DOUBLE
                ) AS age_number
            FROM customers
        )

        SELECT
            count(*) FILTER (
                WHERE raw_age IS NULL
            ) AS blank_ages,

            count(*) FILTER (
                WHERE raw_age IS NOT NULL
                AND (
                    age_number IS NULL
                    OR NOT isfinite(age_number)
                )
            ) AS invalid_or_nonfinite_ages,

            count(*) FILTER (
                WHERE isfinite(age_number)
                AND age_number <> floor(age_number)
            ) AS noninteger_ages,

            count(*) FILTER (
                WHERE isfinite(age_number)
                AND (
                    age_number < 0
                    OR age_number > 120
                )
            ) AS ages_outside_0_to_120,

            min(age_number) FILTER (
                WHERE isfinite(age_number)
            ) AS numeric_min,

            max(age_number) FILTER (
                WHERE isfinite(age_number)
            ) AS numeric_max,

            median(age_number) FILTER (
                WHERE isfinite(age_number)
            ) AS numeric_median

        FROM ages
    """)[0]

    report["age_note"] = (
        "0 to 120 is a plausibility flag for review, "
        "not an automatic deletion rule."
    )

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT,
    )

    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    files = [
        "customers.csv",
        "articles.csv",
        "transactions_train.csv",
    ]

    missing = [
        name
        for name in files
        if not (data_dir / name).is_file()
    ]

    if missing:
        print(
            "Missing files: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    try:
        # The 1 GB setting limits DuckDB's buffer manager,
        # not total process memory.
        # Temporary disk usage is capped at 2 GB.
        with tempfile.TemporaryDirectory(
            prefix=".relationship_tmp_",
            dir=ROOT,
        ) as temp_dir:
            with duckdb.connect(config={
                "memory_limit": "1GB",
                "threads": "2",
                "temp_directory": temp_dir,
                "max_temp_directory_size": "2GB",
            }) as con:
                report = audit(con, data_dir)

        output_dir = ROOT / "reports"
        output_dir.mkdir(exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )

        output = output_dir / (
            f"relationship_check_{stamp}.json"
        )

        with output.open("x", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")

        sections = [
            "customer_ids",
            "article_ids",
            "transaction_links",
            "age_profile",
        ]

        for section in sections:
            print(f"\n{section}:")

            for key, value in report[section].items():
                print(f"  {key}: {value}")

        print(f"\nReport saved: {output}")
        print(
            "Completed. Review the findings before "
            "cleaning or joining the data."
        )
        print(
            "Original CSVs were not changed. "
            "No individual IDs are exported in the report."
        )

        return 0

    except (OSError, duckdb.Error, ValueError) as exc:
        print(
            f"Audit stopped: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)