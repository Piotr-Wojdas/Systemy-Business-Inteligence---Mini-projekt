import csv
import os
from pathlib import Path

import dlt
import holidays
import polars as pl
from dotenv import load_dotenv
from polars import LazyFrame

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

    df: LazyFrame = pl.scan_parquet(file_path)  # LazyFrame

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

    conditions = []

    # datetime sanity (always safe here)
    conditions.append(pl.col("dropoff_datetime") >= pl.col("pickup_datetime"))

    # passenger_count
    if col_exists("passenger_count"):
        conditions.append(pl.col("passenger_count") > 0)

    # trip_distance
    if col_exists("trip_distance"):
        conditions.append(pl.col("trip_distance") > 0)

    # improved_surcharge
    if col_exists("improvement_surcharge"):
        conditions.append(pl.col("improvement_surcharge") >= 0)

    # congestion_surcharge
    if col_exists("congestion_surcharge"):
        conditions.append(pl.col("congestion_surcharge") != 0)

    # cbd_congestion_fee
    if col_exists("cbd_congestion_fee"):
        conditions.append(pl.col("cbd_congestion_fee") != 0)

    # total_amount
    if col_exists("total_amount"):
        conditions.append(pl.col("total_amount") > 0)

    # mta_tax
    if col_exists("mta_tax"):
        conditions.append(pl.col("mta_tax") >= 0)

    # payment_type
    if col_exists("payment_type"):
        conditions.append(~pl.col("payment_type").is_in([5, 6]))

    # PULocationID / DOLocationID invalid zones
    conditions.append(~pl.col("PULocationID").is_in([264, 265]))
    conditions.append(~pl.col("DOLocationID").is_in([264, 265]))

    # apply combined filter
    df = df.filter(pl.reduce(lambda a, b: a & b, conditions))

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

    if category in ("yellow", "green") and "payment_type" in existing_columns:
        df = df.with_columns(
            pl.col("payment_type").cast(pl.Utf8).replace(payment_lookup).alias("payment_type"),
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
                "driver_payout"
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

    df_collected = df.collect(engine="streaming")

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
        full_refresh=True,  # drop empty columns
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
