import csv
import os
from pathlib import Path

import dlt
import holidays
import polars as pl
from dotenv import load_dotenv
from polars import LazyFrame

DATA_DIR = Path("./data/main_2026-01")
LOG_DIR = Path("./logs")
SUMMARY_LOG_FILE = LOG_DIR / "etl_summary.log"
REJECTION_LOG_FILE = LOG_DIR / "rejected_rows.log"
VALIDATION_LOG_FILE = LOG_DIR / "etl_validation.log"

load_dotenv()

os.environ["DESTINATION__POSTGRES__CREDENTIALS"] = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@localhost:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

COLUMNS_TO_DROP = [
    "store_and_fwd_flag",
    "shared_request_flag",
    "dispatching_base_num",
    "originating_base_num",
    "shared_match_flag",
    "wav_request_flag",
    "wav_match_flag",
    "access_a_ride_flag",
    "request_datetime",
    "on_scene_datetime",
    "RatecodeID",
    "trip_type",  # green only
    "trip_time",  # only fhvhv, and includes arrival etc.
    # "hvfhs_license_num", # lift/uber etc
]


def append_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def count_rows(frame: LazyFrame) -> int:
    return frame.select(pl.len()).collect(engine="streaming").item()


def load_lookup_map(path: Path, key_col: str, value_col: str, key_cast=None) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        result = {}
        for row in reader:
            clean_row = {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            key = clean_row[key_col]
            value = clean_row[value_col]
            if key_cast is not None:
                key = key_cast(key)
            result[key] = value
    return result


def process_file(  # noqa: PLR0913, PLR0915
    file_path,
    category,
    pickup_col,
    dropoff_col,
    zones,
    payment_lookup,
    vendor_lookup,
    hvfhs_lookup,
    dl_holidays,
):
    print(f"Extracting: {file_path.name}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df: LazyFrame = pl.scan_parquet(file_path)  # LazyFrame

    input_count = count_rows(df)

    existing_columns = set(df.collect_schema().names())

    # remove columns
    df = df.drop(
        [col for col in COLUMNS_TO_DROP if col in existing_columns],
    )

    df = df.rename(
        {
            pickup_col: "pickup_datetime",
            dropoff_col: "dropoff_datetime",
        },
    ).with_columns(
        pl.lit(category).alias("category"),
        pl.col("PULocationID").cast(pl.Int64),
        pl.col("DOLocationID").cast(pl.Int64),
    )

    if category == "fhvhv" and "trip_miles" in existing_columns:
        df = df.rename({"trip_miles": "trip_distance"})

    existing_columns = set(df.collect_schema().names())

    def col_exists(name: str) -> bool:
        return name in existing_columns

    # ------------------------------------------------------------------
    # DATA QUALITY RULES + REJECTION LOGGING
    # ------------------------------------------------------------------
    rules = []

    # datetime sanity (always safe here)
    rules.append(("invalid_datetime", pl.col("dropoff_datetime") >= pl.col("pickup_datetime")))

    # passenger_count
    if col_exists("passenger_count"):
        rules.append(("invalid_passenger_count", pl.col("passenger_count") > 0))

    # trip_distance
    if col_exists("trip_distance"):
        rules.append(("invalid_trip_distance", pl.col("trip_distance") > 0))

    # improved_surcharge
    if col_exists("improvement_surcharge"):
        rules.append(("invalid_improvement_surcharge", pl.col("improvement_surcharge") >= 0))

    # congestion_surcharge
    if col_exists("congestion_surcharge"):
        rules.append(("invalid_congestion_surcharge", pl.col("congestion_surcharge") != 0))

    # cbd_congestion_fee
    if col_exists("cbd_congestion_fee"):
        rules.append(("invalid_cbd_congestion_fee", pl.col("cbd_congestion_fee") != 0))

    # total_amount
    if col_exists("total_amount"):
        rules.append(("invalid_total_amount", pl.col("total_amount") > 0))

    # mta_tax
    if col_exists("mta_tax"):
        rules.append(("invalid_mta_tax", pl.col("mta_tax") >= 0))

    # payment_type
    if col_exists("payment_type"):
        rules.append(("invalid_payment_type", ~pl.col("payment_type").is_in([5, 6])))

    # PULocationID / DOLocationID invalid zones
    rules.append(("invalid_pickup_zone_id", ~pl.col("PULocationID").is_in([264, 265])))
    rules.append(("invalid_dropoff_zone_id", ~pl.col("DOLocationID").is_in([264, 265])))

    rejection_sample_columns = [
        "pickup_datetime",
        "dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "trip_distance",
        "improvement_surcharge",
        "congestion_surcharge",
        "cbd_congestion_fee",
        "total_amount",
        "mta_tax",
        "payment_type",
        "VendorID",
        "hvfhs_license_num",
        "fare_amount",
        "extra",
        "tolls_amount",
        "tip_amount",
        "base_passenger_fare",
        "tolls",
        "sales_tax",
        "bcf",
        "driver_pay",
        "tips_amount",
    ]
    rejection_sample_columns = [c for c in rejection_sample_columns if c in existing_columns]

    total_rejected_rows = 0

    for rule_name, rule_expr in rules:
        rejected_lf = df.filter(~rule_expr)
        rejected_count = count_rows(rejected_lf)

        if rejected_count > 0:
            total_rejected_rows += rejected_count
            append_log_line(
                REJECTION_LOG_FILE,
                f"{file_path.name};{rule_name};count={rejected_count}",
            )

            rejected_sample = rejected_lf.select(rejection_sample_columns).limit(5).collect(engine="streaming")
            for row in rejected_sample.iter_rows(named=True):  # ty:ignore[unresolved-attribute]
                append_log_line(
                    REJECTION_LOG_FILE,
                    f"{file_path.name};{rule_name};sample={row}",
                )

    # apply combined filter
    df = df.filter(pl.reduce(lambda a, b: a & b, [expr for _, expr in rules]))

    filtered_count = count_rows(df)

    append_log_line(
        SUMMARY_LOG_FILE,
        f"{file_path.name};input={input_count};filtered={filtered_count};removed={input_count - filtered_count}",
    )

    assert filtered_count <= input_count, (
        f"{file_path.name}: filtered count is greater than input count ({filtered_count} > {input_count})"
    )

    df = (
        df.with_columns(
            pl.col("pickup_datetime").dt.date().alias("pickup_date"),
            pl.col("pickup_datetime").dt.time().alias("pickup_time"),
            pl.col("dropoff_datetime").dt.date().alias("dropoff_date"),
            pl.col("dropoff_datetime").dt.time().alias("dropoff_time"),
        )
        .with_columns(
            pl.col("pickup_date").dt.month().alias("month"),
            pl.col("pickup_date").dt.strftime("%A").alias("day_of_week"),
            pl.col("pickup_date").is_in(pl.Series(list(dl_holidays))).alias("is_holiday"),
        )
        .drop(["pickup_datetime", "dropoff_datetime"])
    )

    expected_temporal_columns = {
        "pickup_date",
        "pickup_time",
        "dropoff_date",
        "dropoff_time",
        "month",
        "day_of_week",
        "is_holiday",
    }
    current_schema = set(df.collect_schema().names())
    missing_temporal_columns = expected_temporal_columns - current_schema
    assert not missing_temporal_columns, (
        f"{file_path.name}: missing expected temporal columns: {sorted(missing_temporal_columns)}"
    )

    df = (
        df.join(
            zones,
            left_on="PULocationID",
            right_on="LocationID",
            how="left",
        )
        .rename({"ZoneName": "pickup_zone"})
        .drop("PULocationID")
    )

    df = (
        df.join(
            zones,
            left_on="DOLocationID",
            right_on="LocationID",
            how="left",
        )
        .rename({"ZoneName": "dropoff_zone"})
        .drop("DOLocationID")
    )

    pickup_zone_nulls = df.select(pl.col("pickup_zone").is_null().sum()).collect(engine="streaming").item()
    dropoff_zone_nulls = df.select(pl.col("dropoff_zone").is_null().sum()).collect(engine="streaming").item()

    append_log_line(
        VALIDATION_LOG_FILE,
        f"{file_path.name};pickup_zone_nulls={pickup_zone_nulls};dropoff_zone_nulls={dropoff_zone_nulls}",
    )

    if category in ("yellow", "green") and "payment_type" in existing_columns:
        df = df.with_columns(
            pl.col("payment_type").cast(pl.Utf8).replace(payment_lookup).alias("payment_type"),
        )

        payment_type_nulls = df.select(pl.col("payment_type").is_null().sum()).collect(engine="streaming").item()
        append_log_line(
            VALIDATION_LOG_FILE,
            f"{file_path.name};payment_type_nulls={payment_type_nulls}",
        )

    if category in ("yellow", "green") and "tip_amount" in existing_columns:
        df = df.with_columns(
            pl.when(pl.col("payment_type") == "Credit card").then(pl.col("tip_amount")).otherwise(None).alias("tips"),
        )

    if category == "fhvhv" and "tips_amount" in existing_columns:
        df = df.rename({"tips_amount": "tips"})

    vendor_exprs = []

    if category in ("yellow", "green") and "VendorID" in existing_columns:
        vendor_exprs.append(pl.col("VendorID").cast(pl.Utf8).replace(vendor_lookup))

    if category == "fhvhv" and "hvfhs_license_num" in existing_columns:
        vendor_exprs.append(pl.col("hvfhs_license_num").cast(pl.Utf8).replace(hvfhs_lookup))

    if vendor_exprs:
        df = df.with_columns(pl.coalesce(vendor_exprs).alias("vendor")).drop(
            [c for c in ["VendorID", "hvfhs_license_num"] if c in df.collect_schema().names()],
        )

        vendor_nulls = df.select(pl.col("vendor").is_null().sum()).collect(engine="streaming").item()
        append_log_line(
            VALIDATION_LOG_FILE,
            f"{file_path.name};vendor_nulls={vendor_nulls}",
        )

    if category == "yellow":
        df = df.with_columns(
            (pl.col("fare_amount").fill_null(0) + pl.col("extra").fill_null(0)).alias("base_fare"),
            (
                pl.col("tolls_amount").fill_null(0)
                + pl.col("congestion_surcharge").fill_null(0)
                + pl.col("cbd_congestion_fee").fill_null(0)
                + pl.col("Airport_fee").fill_null(0)
            ).alias("tolls_and_fees"),
            (pl.col("mta_tax").fill_null(0) + pl.col("improvement_surcharge").fill_null(0)).alias("taxes"),
            pl.col("tip_amount").alias("tips"),
            pl.col("total_amount").alias("total_passenger_paid"),
            (
                pl.col("fare_amount").fill_null(0)
                + pl.col("extra").fill_null(0)
                + pl.col("tolls_amount").fill_null(0)
                + pl.col("tip_amount").fill_null(0)
            ).alias("driver_payout"),
        ).drop(
            [
                c
                for c in [
                    "fare_amount",
                    "extra",
                    "tolls_amount",
                    "congestion_surcharge",
                    "cbd_congestion_fee",
                    "Airport_fee",
                    "mta_tax",
                    "improvement_surcharge",
                    "tip_amount",
                    "total_amount",
                ]
                if c in df.collect_schema().names()
            ],
        )

    if category == "green":
        df = df.with_columns(
            (pl.col("fare_amount").fill_null(0) + pl.col("extra").fill_null(0)).alias("base_fare"),
            (
                pl.col("tolls_amount").fill_null(0)
                + pl.col("congestion_surcharge").fill_null(0)
                + pl.col("cbd_congestion_fee").fill_null(0)
            ).alias("tolls_and_fees"),
            (pl.col("mta_tax").fill_null(0) + pl.col("improvement_surcharge").fill_null(0)).alias("taxes"),
            pl.col("tip_amount").alias("tips"),
            pl.col("total_amount").alias("total_passenger_paid"),
            (
                pl.col("fare_amount").fill_null(0)
                + pl.col("extra").fill_null(0)
                + pl.col("tolls_amount").fill_null(0)
                + pl.col("tip_amount").fill_null(0)
            ).alias("driver_payout"),
        ).drop(
            [
                c
                for c in [
                    "fare_amount",
                    "extra",
                    "tolls_amount",
                    "congestion_surcharge",
                    "cbd_congestion_fee",
                    "mta_tax",
                    "improvement_surcharge",
                    "tip_amount",
                    "total_amount",
                ]
                if c in df.collect_schema().names()
            ],
        )

    if category == "fhvhv":
        df = df.with_columns(
            pl.col("base_passenger_fare").fill_null(0).alias("base_fare"),
            (
                pl.col("tolls").fill_null(0)
                + pl.col("congestion_surcharge").fill_null(0)
                + pl.col("cbd_congestion_fee").fill_null(0)
                + pl.col("airport_fee").fill_null(0)
            ).alias("tolls_and_fees"),
            (pl.col("sales_tax").fill_null(0) + pl.col("bcf").fill_null(0)).alias("taxes"),
            (
                pl.col("base_passenger_fare").fill_null(0)
                + pl.col("tolls").fill_null(0)
                + pl.col("congestion_surcharge").fill_null(0)
                + pl.col("cbd_congestion_fee").fill_null(0)
                + pl.col("airport_fee").fill_null(0)
                + pl.col("sales_tax").fill_null(0)
                + pl.col("bcf").fill_null(0)
                + pl.col("tips").fill_null(0)
            ).alias("total_passenger_paid"),
            (pl.col("driver_pay").fill_null(0) + pl.col("tolls").fill_null(0) + pl.col("tips").fill_null(0)).alias(
                "driver_payout",
            ),
        ).drop(
            [
                c
                for c in [
                    "base_passenger_fare",
                    "tolls",
                    "congestion_surcharge",
                    "cbd_congestion_fee",
                    "airport_fee",
                    "sales_tax",
                    "bcf",
                ]
                if c in df.collect_schema().names()
            ],
        )

    final_schema = set(df.collect_schema().names())
    if "pickup_zone" in final_schema and "dropoff_zone" in final_schema:
        zone_validation_count = (
            df.select(
                (pl.col("pickup_zone").is_null() | pl.col("dropoff_zone").is_null()).sum(),
            )
            .collect(engine="streaming")
            .item()
        )
        append_log_line(
            VALIDATION_LOG_FILE,
            f"{file_path.name};zone_validation_null_rows={zone_validation_count}",
        )

    df_collected = df.collect(engine="streaming")

    assert df_collected.height == filtered_count, (
        f"{file_path.name}: row count changed after transformations ({df_collected.height} != {filtered_count})"
    )

    for batch in df_collected.iter_slices(n_rows=10000):  # ty:ignore[unresolved-attribute]
        yield batch.to_arrow().to_pylist()  # eugh, ale równolegle i tak jest szybciej


def get_taxi_resources():
    data_dir = Path("./data")
    zone_file = data_dir / "lookup/taxi_zone_lookup.csv"

    zone_file = data_dir / "lookup/taxi_zone_lookup.csv"
    payment_file = data_dir / "lookup/paymenttype.csv"
    vendor_file = data_dir / "lookup/vendorid.csv"

    zones_lazy = pl.scan_csv(zone_file).select(
        pl.col("LocationID").cast(pl.Int64),
        pl.col("Zone").alias("ZoneName"),
    )

    payment_lookup = {
        str(k): v
        for k, v in load_lookup_map(
            payment_file,
            "payment_type_code",
            "payment_type_string",
            key_cast=int,
        ).items()
    }

    vendor_lookup = {
        str(k): v
        for k, v in load_lookup_map(
            vendor_file,
            "vendor_id_code",
            "vendor_id",
            key_cast=int,
        ).items()
    }

    hvfhs_lookup = {
        "HV0002": "Juno",
        "HV0003": "Uber",
        "HV0004": "Via",
        "HV0005": "Lyft",
    }

    dl_holidays = holidays.US(years=range(2015, 2027))

    def create_resource(file_path, category, p_col, d_col):
        @dlt.resource(name=file_path.stem, table_name="imported", write_disposition="append")
        def resource_generator():
            yield from process_file(
                file_path,
                category,
                p_col,
                d_col,
                zones_lazy,
                payment_lookup,
                vendor_lookup,
                hvfhs_lookup,
                dl_holidays,
            )

        return resource_generator()

    return (
        [create_resource(f, "yellow", "tpep_pickup_datetime", "tpep_dropoff_datetime") for f in data_dir.glob("yellow*.parquet")] +
        [create_resource(f, "green", "lpep_pickup_datetime", "lpep_dropoff_datetime") for f in data_dir.glob("green*.parquet")] +
        [create_resource(f, "fhvhv", "pickup_datetime", "dropoff_datetime") for f in data_dir.glob("fhvhv*.parquet")]
    )  # fmt: skip


def main():
    pipeline = dlt.pipeline(
        pipeline_name="taxi_pipeline",
        destination="postgres",
        dataset_name="raw",
        progress="enlighten",
        dev_mode=True,
        # full_refresh=True,  # drop empty columns
    )

    print("Extraction...")
    pipeline.extract(get_taxi_resources(), workers=8)

    print("Normalization...")
    pipeline.normalize()

    print("Loading to postgres...")
    load_info = pipeline.load()

    print(load_info)


if __name__ == "__main__":
    main()
