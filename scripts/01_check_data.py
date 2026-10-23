"""Read-only H&M data check. Quick sample by default; add --full for a scan."""

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "transactions_train.csv": [
        "t_dat", "customer_id", "article_id", "price", "sales_channel_id"
    ],
    "customers.csv": [
        "customer_id", "FN", "Active", "club_member_status",
        "fashion_news_frequency", "age", "postal_code"
    ],
    "articles.csv": [
        "article_id", "product_code", "prod_name", "product_type_no",
        "product_type_name", "product_group_name", "graphical_appearance_no",
        "graphical_appearance_name", "colour_group_code", "colour_group_name",
        "perceived_colour_value_id", "perceived_colour_value_name",
        "perceived_colour_master_id", "perceived_colour_master_name",
        "department_no", "department_name", "index_code", "index_name",
        "index_group_no", "index_group_name", "section_no", "section_name",
        "garment_group_no", "garment_group_name", "detail_desc"
    ],
}


def quick_check(path, expected, limit):
    """Read a small sample, preserving all values as strings."""
    result = {"size_bytes": path.stat().st_size}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        columns = next(reader, [])

        result["columns"] = columns
        result["missing_columns"] = sorted(set(expected) - set(columns))
        result["unexpected_columns"] = sorted(set(columns) - set(expected))

        if (
            not columns
            or result["missing_columns"]
            or len(set(columns)) != len(columns)
        ):
            raise ValueError(
                f"{path.name}: unexpected header. Check the original CSV."
            )

        blanks = dict.fromkeys(columns, 0)
        rows = 0

        for number, record in enumerate(islice(reader, limit), start=2):
            if len(record) != len(columns):
                raise ValueError(
                    f"{path.name}: wrong field count at record {number}."
                )

            rows += 1

            for column, value in zip(columns, record):
                blanks[column] += not value.strip()

        if rows == 0:
            raise ValueError(f"{path.name}: contains no data records.")

    result["sample_rows_read"] = rows
    result["sample_blank_values"] = blanks

    return result


def full_check(connection, path, columns):
    """Query full-file aggregates without loading all rows into Python."""
    expressions = ["count(*) AS row_count"]

    for column in columns:
        identifier = '"' + column.replace('"', '""') + '"'
        alias = '"blank_' + column.replace('"', '""') + '"'

        expressions.append(
            f"count(*) FILTER "
            f"(WHERE nullif(trim({identifier}), '') IS NULL) AS {alias}"
        )

    if path.name == "transactions_train.csv":
        expressions += [
            "min(try_cast(t_dat AS DATE)) AS first_transaction_date",
            "max(try_cast(t_dat AS DATE)) AS last_transaction_date",
            "count(*) FILTER "
            "(WHERE nullif(trim(t_dat), '') IS NOT NULL "
            "AND try_cast(t_dat AS DATE) IS NULL) AS invalid_date_values",
            "count(*) FILTER "
            "(WHERE nullif(trim(price), '') IS NOT NULL "
            "AND (try_cast(price AS DOUBLE) IS NULL "
            "OR NOT isfinite(try_cast(price AS DOUBLE)))) "
            "AS invalid_price_values",
            "count(*) FILTER "
            "(WHERE try_cast(price AS DOUBLE) < 0) AS negative_price_values",
            "count(*) FILTER "
            "(WHERE nullif(trim(sales_channel_id), '') IS NOT NULL "
            "AND trim(sales_channel_id) NOT IN ('1', '2')) "
            "AS unexpected_channel_values",
        ]

    # Parameter binding handles spaces and quotes in folder paths.
    # Reading as VARCHAR preserves leading zeroes in identifiers.
    query = "SELECT " + ", ".join(expressions) + """
        FROM read_csv(
            ?,
            header=true,
            delim=',',
            quote='"',
            escape='"',
            all_varchar=true,
            strict_mode=true,
            ignore_errors=false
        )
    """

    cursor = connection.execute(query, [str(path)])
    names = [entry[0] for entry in cursor.description]

    return dict(zip(names, cursor.fetchone()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        help="Folder containing all three extracted CSVs",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also scan every record for aggregate checks",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1000,
        help="Quick sample size; default is 1000",
    )

    args = parser.parse_args()

    if args.sample_rows < 1:
        parser.error("--sample-rows must be at least 1")

    try:
        import duckdb
    except ImportError:
        print(
            "DuckDB is missing. Install it using your .venv Python.",
            file=sys.stderr,
        )
        return 1

    folder = args.data_dir

    if not folder:
        print(
            "Paste the folder path containing transactions_train.csv, "
            "customers.csv and articles.csv."
        )

        try:
            folder = input("CSV folder: ").strip()
        except EOFError:
            print(
                "Use --data-dir with the CSV folder path.",
                file=sys.stderr,
            )
            return 1

    folder = folder.strip()

    if (
        len(folder) >= 2
        and folder[0] == folder[-1]
        and folder[0] in ('"', "'")
    ):
        folder = folder[1:-1]

    if not folder:
        print("Please enter a folder path.", file=sys.stderr)
        return 1

    data_dir = Path(folder).expanduser().resolve()

    missing = [
        name
        for name in REQUIRED
        if not (data_dir / name).is_file()
    ]

    if missing:
        print("Missing files: " + ", ".join(missing), file=sys.stderr)
        print(
            "Select the folder containing the three extracted CSVs.",
            file=sys.stderr,
        )
        return 1

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "duckdb_version": duckdb.__version__,
        "mode": "full" if args.full else "quick_sample",
        "sample_rows_requested": args.sample_rows,
        "scope": (
            "Headers, first records, full-file row and blank counts, "
            "and transaction date, price and channel checks. "
            "ID uniqueness and relationships are not checked."
            if args.full
            else
            "Headers and first records only. "
            "Not a random sample or a full-file quality audit."
        ),
        "files": {},
    }

    try:
        for name, expected_columns in REQUIRED.items():
            result = quick_check(
                data_dir / name,
                expected_columns,
                args.sample_rows,
            )

            report["files"][name] = result

            size_mb = result["size_bytes"] / 1_000_000

            print(
                f"CHECKED {name}: {size_mb:,.1f} MB; "
                f"read {result['sample_rows_read']:,} sample rows",
                flush=True,
            )

            if result["unexpected_columns"]:
                print(
                    "  Additional columns: "
                    + ", ".join(result["unexpected_columns"])
                )

        if args.full:
            # This limits DuckDB's buffer manager, not total process memory.
            with duckdb.connect(
                config={"memory_limit": "1GB", "threads": "2"}
            ) as connection:
                for name, result in report["files"].items():
                    print(
                        f"SCANNING {name} ... "
                        "this may take several minutes.",
                        flush=True,
                    )

                    result["full_scan"] = full_check(
                        connection,
                        data_dir / name,
                        result["columns"],
                    )

                    print(
                        "  Full row count: "
                        f"{result['full_scan']['row_count']:,}",
                        flush=True,
                    )

                stats = report["files"]["transactions_train.csv"]["full_scan"]

                print(
                    "Transaction date range: "
                    f"{stats['first_transaction_date']} to "
                    f"{stats['last_transaction_date']}"
                )

        report_dir = ROOT / "reports"
        report_dir.mkdir(exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )

        output = report_dir / (
            f"data_check_{report['mode']}_{stamp}.json"
        )

        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
            handle.write("\n")

        print(f"\nReport saved: {output}")
        print(
            "The report contains aggregate checks, "
            "not customer IDs or individual records."
        )
        print(
            "Check completed. Review the report; "
            "completion does not mean all values are valid."
        )

        if not args.full:
            print(
                "Only the first records were read. "
                "Add --full for complete row counts and date coverage."
            )

        return 0

    except (OSError, ValueError, csv.Error, duckdb.Error) as exc:
        print(f"Check stopped: {exc}", file=sys.stderr)
        print(
            "No successful report was written for this run. "
            "The original CSVs were not modified.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)