import os
from pathlib import Path

import dlt
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
    "hvfhs_license_num",
]


def process_file(file_path, category, pickup_col, dropoff_col, zones):
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
    if col_exists("improved_surcharge"):
        conditions.append(pl.col("improved_surcharge") >= 0)

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

    df = df.with_columns(
        pl.col("pickup_datetime").dt.date().alias("pickup_date"),
        pl.col("pickup_datetime").dt.time().alias("pickup_time"),
        pl.col("dropoff_datetime").dt.date().alias("dropoff_date"),
        pl.col("dropoff_datetime").dt.time().alias("dropoff_time"),
    ).drop(["pickup_datetime", "dropoff_datetime"])

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

    df_collected = df.collect(engine="streaming")

    for batch in df_collected.iter_slices(n_rows=10000):  # ty:ignore[unresolved-attribute]
        yield batch.to_arrow().to_pylist()  # eugh, ale równolegle i tak jest szybciej


def get_taxi_resources():
    data_dir = Path("./data")
    zone_file = data_dir / "lookup/taxi_zone_lookup.csv"

    zones_lazy = pl.scan_csv(zone_file).select(
        pl.col("LocationID").cast(pl.Int64),
        pl.col("Zone").alias("ZoneName"),
    )

    def create_resource(file_path, category, p_col, d_col):
        @dlt.resource(name=file_path.stem, table_name="staging", write_disposition="append")
        def resource_generator():
            yield from process_file(file_path, category, p_col, d_col, zones_lazy)

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
