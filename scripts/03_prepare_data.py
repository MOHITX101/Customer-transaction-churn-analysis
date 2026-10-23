"""Create verified Parquet files without changing the source CSVs."""

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3

CSV_SOURCE = """read_csv(
    ?, header=true, delim=',', quote='"', escape='"',
    all_varchar=true, strict_mode=true, ignore_errors=false
)"""

CUSTOMERS_SQL = """
    SELECT
        customer_id,

        CASE
            WHEN nullif(trim(FN), '') IS NULL THEN NULL
            WHEN trim(FN) IN ('1', '1.0') THEN 1::UTINYINT
            WHEN trim(FN) IN ('0', '0.0') THEN 0::UTINYINT
            ELSE error('Unexpected FN value')
        END AS FN,

        CASE
            WHEN nullif(trim(Active), '') IS NULL THEN NULL
            WHEN trim(Active) IN ('1', '1.0') THEN 1::UTINYINT
            WHEN trim(Active) IN ('0', '0.0') THEN 0::UTINYINT
            ELSE error('Unexpected Active value')
        END AS Active,

        upper(
            nullif(trim(club_member_status), '')
        ) AS club_member_status,

        upper(
            nullif(trim(fashion_news_frequency), '')
        ) AS fashion_news_frequency,

        CASE
            WHEN nullif(trim(age), '') IS NULL THEN NULL
            WHEN isfinite(try_cast(age AS DOUBLE))
                AND try_cast(age AS DOUBLE) BETWEEN 0 AND 120
                AND try_cast(age AS DOUBLE)
                    = floor(try_cast(age AS DOUBLE))
            THEN cast(age AS SMALLINT)
            ELSE error('Unexpected age; review the age audit')
        END AS age,

        postal_code,

        nullif(trim(age), '') IS NULL AS age_missing,
        nullif(trim(FN), '') IS NULL AS fn_missing,
        nullif(trim(Active), '') IS NULL AS active_missing

    FROM """ + CSV_SOURCE

ARTICLES_SQL = """
    SELECT
        *,
        nullif(trim(detail_desc), '') IS NULL
            AS detail_desc_missing

    FROM """ + CSV_SOURCE

TRANSACTIONS_SQL = """
    SELECT
        cast(t_dat AS DATE) AS t_dat,
        customer_id,
        article_id,

        CASE
            WHEN isfinite(cast(price AS DOUBLE))
                AND cast(price AS DOUBLE) >= 0
            THEN cast(price AS DOUBLE)
            ELSE error(
                'Invalid price; review the transaction audit'
            )
        END AS price,

        CASE
            WHEN trim(sales_channel_id) IN ('1', '2')
            THEN cast(sales_channel_id AS UTINYINT)
            ELSE error('Unexpected sales channel')
        END AS sales_channel_id

    FROM """ + CSV_SOURCE

SPECS = [
    ("customers.csv", "customers.parquet", CUSTOMERS_SQL),
    ("articles.csv", "articles.parquet", ARTICLES_SQL),
    (
        "transactions_train.csv",
        "transactions.parquet",
        TRANSACTIONS_SQL,
    ),
]


def sql_string(value):
    # Escape quotes in COPY output paths.
    return "'" + str(value).replace("'", "''") + "'"


def one_row(con, sql, parameters):
    cursor = con.execute(sql, parameters)
    names = [item[0] for item in cursor.description]

    return dict(zip(names, cursor.fetchone()))


def prepare(data_dir, data_parent):
    target = data_parent / "processed"

    if target.exists():
        raise ValueError(
            f"Output folder already exists: {target}. "
            "Existing outputs were not overwritten."
        )

    sources = [
        data_dir / spec[0]
        for spec in SPECS
    ]

    missing = [
        path.name
        for path in sources
        if not path.is_file()
    ]

    if missing:
        raise ValueError(
            "Missing files: " + ", ".join(missing)
        )

    data_parent.mkdir(parents=True, exist_ok=True)

    source_bytes = sum(
        path.stat().st_size
        for path in sources
    )

    required_free = source_bytes + 2 * GIB
    free = shutil.disk_usage(data_parent).free

    print(
        f"Free disk space: {free / GIB:.2f} GiB",
        flush=True,
    )
    print(
        "Conservative starting-space check: "
        f"{required_free / GIB:.2f} GiB",
        flush=True,
    )

    if free < required_free:
        raise ValueError(
            "Not enough free space for this conversion. "
            "CSVs are unchanged; send this message "
            "for the next step."
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "duckdb_version": duckdb.__version__,
        "compression": "ZSTD",
        "row_group_size": 65536,
        "price_units": (
            "Original dataset units; no currency conversion "
            "or scaling applied."
        ),
        "cleaning_rules": [
            "IDs, postal codes, and article category codes remain text.",
            "FN and Active become nullable 0/1 integers; blanks remain unknown.",
            "Customer category labels are trimmed and uppercased; blank values remain null.",
            "Missing ages remain null; missing-value indicators are added.",
            "Dates and transaction numbers receive explicit types.",
            "Article text is retained, with a description-missing indicator.",
            "All transaction rows are preserved, including repeated purchases.",
            "No training features, imputation, or churn labels are created.",
        ],
        "files": {},
    }

    with tempfile.TemporaryDirectory(
        prefix="prepare_run_",
        dir=data_parent,
    ) as work:
        stage = Path(work) / "processed"
        stage.mkdir()

        # The memory setting limits DuckDB's buffer manager,
        # not total process memory.
        with duckdb.connect(config={
            "memory_limit": "1GB",
            "threads": "2",
            "temp_directory": str(Path(work) / "spill"),
            "max_temp_directory_size": "1GB",
            "preserve_insertion_order": "false",
        }) as con:

            for source_name, output_name, select_sql in SPECS:
                source = data_dir / source_name

                signature = (
                    source.stat().st_size,
                    source.stat().st_mtime_ns,
                )

                print(
                    f"\nCounting source rows: {source_name} ...",
                    flush=True,
                )

                source_rows = con.execute(
                    "SELECT count(*) FROM " + CSV_SOURCE,
                    [str(source)],
                ).fetchone()[0]

                if source_rows == 0:
                    raise ValueError(
                        f"{source_name} contains no data rows."
                    )

                output = stage / output_name

                print(
                    f"Writing {output_name} ...",
                    flush=True,
                )

                con.execute(
                    f"COPY ({select_sql}) "
                    f"TO {sql_string(output)} "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, "
                    "ROW_GROUP_SIZE 65536)",
                    [str(source)],
                )

                parquet_rows = con.execute(
                    "SELECT count(*) FROM read_parquet(?)",
                    [str(output)],
                ).fetchone()[0]

                if parquet_rows != source_rows:
                    raise ValueError(
                        f"Row-count mismatch for {source_name}."
                    )

                current_signature = (
                    source.stat().st_size,
                    source.stat().st_mtime_ns,
                )

                if signature != current_signature:
                    raise ValueError(
                        f"{source_name} changed during conversion; "
                        "no outputs published."
                    )

                schema = con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [str(output)],
                ).fetchall()

                modified_at = datetime.fromtimestamp(
                    source.stat().st_mtime,
                    timezone.utc,
                ).isoformat()

                report["files"][output_name] = {
                    "source_file": source_name,
                    "source_bytes": signature[0],
                    "source_modified_at_utc": modified_at,
                    "source_rows": source_rows,
                    "parquet_rows": parquet_rows,
                    "row_count_verified": True,
                    "parquet_bytes": output.stat().st_size,
                    "schema": [
                        {
                            "column": item[0],
                            "type": item[1],
                        }
                        for item in schema
                    ],
                }

                print(
                    f"Verified {parquet_rows:,} rows; "
                    f"{output.stat().st_size / GIB:.3f} GiB",
                    flush=True,
                )

            report["customer_missing_values"] = one_row(
                con,
                """
                    SELECT
                        count(*) FILTER (
                            WHERE FN IS NULL
                        ) AS missing_fn,

                        count(*) FILTER (
                            WHERE Active IS NULL
                        ) AS missing_active,

                        count(*) FILTER (
                            WHERE age IS NULL
                        ) AS missing_age,

                        count(*) FILTER (
                            WHERE fashion_news_frequency IS NULL
                        ) AS missing_news_frequency

                    FROM read_parquet(?)
                """,
                [str(stage / "customers.parquet")],
            )

            report["transaction_summary"] = one_row(
                con,
                """
                    SELECT
                        min(t_dat) AS first_date,
                        max(t_dat) AS last_date,

                        count(*) FILTER (
                            WHERE t_dat IS NULL
                                OR customer_id IS NULL
                                OR article_id IS NULL
                                OR price IS NULL
                                OR sales_channel_id IS NULL
                        ) AS rows_with_null_required_fields

                    FROM read_parquet(?)
                """,
                [str(stage / "transactions.parquet")],
            )

            null_rows = report["transaction_summary"][
                "rows_with_null_required_fields"
            ]

            if null_rows:
                raise ValueError(
                    "Prepared transactions contain missing "
                    "required fields; review source data."
                )

        report["total_source_bytes"] = source_bytes

        report["total_parquet_bytes"] = sum(
            item["parquet_bytes"]
            for item in report["files"].values()
        )

        report_path = stage / "preparation_report.json"

        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                default=str,
                allow_nan=False,
            )
            handle.write("\n")

        if target.exists():
            raise ValueError(
                "Output folder appeared during conversion; "
                "existing files were not replaced."
            )

        # Publish the complete folder after successful verification.
        os.rename(stage, target)

    print(f"\nPrepared files: {target}")
    print(f"Report: {target / 'preparation_report.json'}")
    print(
        "Total Parquet size: "
        f"{report['total_parquet_bytes'] / GIB:.3f} GiB"
    )
    print("Conversion complete. Original CSVs are unchanged.")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT,
    )

    args = parser.parse_args()

    try:
        prepare(
            args.data_dir.expanduser().resolve(),
            ROOT / "data",
        )
        return 0

    except (OSError, ValueError, duckdb.Error) as exc:
        print(
            f"Preparation stopped: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nStopped by user. "
            "Temporary conversion files are cleaned up.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    