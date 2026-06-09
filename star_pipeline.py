import os
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import dlt
import holidays
import polars as pl
from dlt.sources.sql_database import sql_table
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# it doesnt work normally for some reason, idk
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")

PG_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"
SQLALCHEMY_URL = PG_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

os.environ["DESTINATION__POSTGRES__CREDENTIALS"] = PG_URL

WEATHER_CSV = Path("data/thirdparty/weather.csv")

olap_pipeline = dlt.pipeline(
    pipeline_name="taxi_star_schema",
    destination="postgres",
    dataset_name="olap",
    progress="enlighten",
)


def detect_latest_raw_schema() -> str:
    engine = create_engine(SQLALCHEMY_URL)
    query = text(
        """
        SELECT table_schema
        FROM information_schema.tables
        WHERE table_name = :table_name
          AND table_schema LIKE :schema_prefix
        ORDER BY table_schema DESC
        LIMIT 1
        """,
    )

    with engine.connect() as conn:
        row = conn.execute(
            query,
            {"table_name": "imported", "schema_prefix": "raw_%"},
        ).fetchone()

    if not row:
        msg = "Could not find table 'imported' in any schema starting with 'raw'."
        raise RuntimeError(msg)

    return row[0]


def date_key(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def time_key(t: time) -> int:
    return t.hour * 10000 + t.minute * 100 + t.second


def part_of_day(t: time) -> str:
    if 5 <= t.hour <= 11:
        return "Morning"
    if 12 <= t.hour <= 16:
        return "Afternoon"
    if 17 <= t.hour <= 21:
        return "Evening"
    return "Night"


def norm_date(v: Any) -> date | None:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        return pl.Series([v]).cast(pl.Date).item()
    except Exception:  # noqa: BLE001
        return None


def norm_time(v: Any) -> time | None:
    if v is None:
        return None
    if isinstance(v, time):
        return v
    try:
        return pl.Series([v]).cast(pl.Time).item()
    except Exception:  # noqa: BLE001
        return None


def raw_source_relation(source_schema: str):
    return sql_table(
        credentials=PG_URL,
        schema=source_schema,
        table="imported",
        backend="pyarrow",
        chunk_size=20000,
        reflection_level="full",
    )


def iter_source_frames(source_schema: str):
    source = raw_source_relation(source_schema)
    for batch in source:
        yield pl.from_arrow(batch)


def build_dimension_maps(source_schema: str):
    min_date = None
    max_date = None

    unique_times = set()
    unique_zones = set()
    unique_vendors = set()
    unique_payment_types = set()
    unique_categories = set()

    for df in iter_source_frames(source_schema):
        if df.is_empty():
            continue

        pickup_dates = df.get_column("pickup_date").drop_nulls().to_list()
        dropoff_dates = df.get_column("dropoff_date").drop_nulls().to_list()

        for v in pickup_dates + dropoff_dates:
            dv = norm_date(v)
            if dv is None:
                continue
            min_date = dv if min_date is None else min(min_date, dv)
            max_date = dv if max_date is None else max(max_date, dv)

        unique_times.update(filter(None, (norm_time(v) for v in df.get_column("pickup_time").drop_nulls().to_list())))
        unique_times.update(filter(None, (norm_time(v) for v in df.get_column("dropoff_time").drop_nulls().to_list())))

        unique_zones.update(str(v) for v in df.get_column("pickup_zone").drop_nulls().to_list() if v is not None)
        unique_zones.update(str(v) for v in df.get_column("dropoff_zone").drop_nulls().to_list() if v is not None)
        unique_vendors.update(str(v) for v in df.get_column("vendor").drop_nulls().to_list() if v is not None)
        unique_payment_types.update(
            str(v) for v in df.get_column("payment_type").drop_nulls().to_list() if v is not None
        )
        unique_categories.update(str(v) for v in df.get_column("category").drop_nulls().to_list() if v is not None)

    if min_date is None or max_date is None:
        msg = "Source table is empty or date columns are missing."
        raise RuntimeError(msg)

    zone_list = sorted(unique_zones)
    vendor_list = sorted(unique_vendors)
    payment_type_list = sorted(unique_payment_types)
    category_list = sorted(unique_categories)
    time_list = sorted(unique_times, key=lambda t: (t.hour, t.minute, t.second))

    holiday_set = holidays.US(years=range(min_date.year, max_date.year + 1))

    return {
        "min_date": min_date,
        "max_date": max_date,
        "holiday_set": holiday_set,
        "zone_list": zone_list,
        "vendor_list": vendor_list,
        "payment_type_list": payment_type_list,
        "category_list": category_list,
        "time_list": time_list,
        "dim_zone_map": {name: idx + 1 for idx, name in enumerate(zone_list)},
        "dim_vendor_map": {name: idx + 1 for idx, name in enumerate(vendor_list)},
        "dim_payment_map": {name: idx + 1 for idx, name in enumerate(payment_type_list)},
        "dim_category_map": {name: idx + 1 for idx, name in enumerate(category_list)},
    }


def build_weather_dimension():
    if not WEATHER_CSV.exists():
        msg = f"Weather file not found: {WEATHER_CSV}"
        raise RuntimeError(msg)

    weather_df = (
        pl.read_csv(WEATHER_CSV, try_parse_dates=True)
        .rename(
            {
                "Zone": "zone_name",
                "date": "weather_date",
                "hour": "weather_hour",
                "temperature_f": "temperature_f",
                "precipitation_inches": "precipitation_inches",
                "snowfall_inches": "snowfall_inches",
                "weather_status": "weather_status",
            },
        )
        .with_columns(
            [
                pl.col("weather_date").cast(pl.Date),
                pl.col("weather_hour").cast(pl.Int64),
                pl.col("temperature_f").cast(pl.Float64),
                ((pl.col("temperature_f") - 32) * 5 / 9).round(2).alias("temperature_c"),
                pl.col("precipitation_inches").cast(pl.Float64),
                pl.col("snowfall_inches").cast(pl.Float64),
                pl.col("weather_status").cast(pl.Utf8),
                pl.col("zone_name").cast(pl.Utf8),
            ],
        )
        .unique(subset=["zone_name", "weather_date", "weather_hour"], keep="first")
        .sort(["zone_name", "weather_date", "weather_hour"])
        .with_row_index("weather_key", offset=1)
        .select(
            [
                "weather_key",
                "zone_name",
                "weather_date",
                "weather_hour",
                "temperature_f",
                "temperature_c",
                "precipitation_inches",
                "snowfall_inches",
                "weather_status",
            ],
        )
    )

    weather_lookup = {
        (row["zone_name"], row["weather_date"], row["weather_hour"]): row["weather_key"]
        for row in weather_df.iter_rows(named=True)
    }

    return weather_df, weather_lookup


def make_dim_date_table(min_date: date, max_date: date, holiday_set: set[date]) -> pl.DataFrame:
    rows = []
    cur = min_date
    while cur <= max_date:
        rows.append(
            {
                "date_key": date_key(cur),
                "full_date": cur,
                "year": cur.year,
                "quarter": (cur.month - 1) // 3 + 1,
                "month": cur.month,
                "month_name": cur.strftime("%B"),
                "day_of_month": cur.day,
                "day_of_week": cur.isoweekday(),
                "day_of_week_name": cur.strftime("%A"),
                "is_weekend": cur.isoweekday() in (6, 7),
                "is_holiday": cur in holiday_set,
            },
        )
        cur += timedelta(days=1)
    return pl.DataFrame(rows)


def make_dim_time_table(time_list: list[time]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_key": [time_key(t) for t in time_list],
            "full_time": time_list,
            "hour": [t.hour for t in time_list],
            "minute": [t.minute for t in time_list],
            "second": [t.second for t in time_list],
            "part_of_day": [part_of_day(t) for t in time_list],
        },
    )


def make_dim_lookup_table(names: list[str], key_name: str, value_name: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            key_name: list(range(1, len(names) + 1)),
            value_name: names,
        },
    )


def make_fact_batch(  # noqa: PLR0913
    df: pl.DataFrame,
    dim_zone_map: dict[str, int],
    dim_vendor_map: dict[str, int],
    dim_payment_map: dict[str, int],
    dim_category_map: dict[str, int],
    weather_lookup: dict[tuple[str, date, int], int],
) -> pl.DataFrame:
    pickup_dt = pl.datetime(
        pl.col("pickup_date").dt.year(),
        pl.col("pickup_date").dt.month(),
        pl.col("pickup_date").dt.day(),
        pl.col("pickup_time").dt.hour(),
        pl.col("pickup_time").dt.minute(),
        pl.col("pickup_time").dt.second(),
    )
    dropoff_dt = pl.datetime(
        pl.col("dropoff_date").dt.year(),
        pl.col("dropoff_date").dt.month(),
        pl.col("dropoff_date").dt.day(),
        pl.col("dropoff_time").dt.hour(),
        pl.col("dropoff_time").dt.minute(),
        pl.col("dropoff_time").dt.second(),
    )

    return df.with_columns(
        [
            (
                pl.col("pickup_date").dt.year() * 10000
                + pl.col("pickup_date").dt.month() * 100
                + pl.col("pickup_date").dt.day()
            ).alias("pickup_date_key"),
            (
                pl.col("dropoff_date").dt.year() * 10000
                + pl.col("dropoff_date").dt.month() * 100
                + pl.col("dropoff_date").dt.day()
            ).alias("dropoff_date_key"),
            (
                pl.col("pickup_time").dt.hour() * 10000
                + pl.col("pickup_time").dt.minute() * 100
                + pl.col("pickup_time").dt.second()
            ).alias("pickup_time_key"),
            (
                pl.col("dropoff_time").dt.hour() * 10000
                + pl.col("dropoff_time").dt.minute() * 100
                + pl.col("dropoff_time").dt.second()
            ).alias("dropoff_time_key"),
            pl.col("pickup_zone").cast(pl.Utf8).replace(dim_zone_map).cast(pl.Int64).alias("pickup_zone_key"),
            pl.col("dropoff_zone").cast(pl.Utf8).replace(dim_zone_map).cast(pl.Int64).alias("dropoff_zone_key"),
            pl.col("vendor").cast(pl.Utf8).replace(dim_vendor_map).cast(pl.Int64).alias("vendor_key"),
            pl.col("payment_type").cast(pl.Utf8).replace(dim_payment_map).cast(pl.Int64).alias("payment_type_key"),
            pl.col("category").cast(pl.Utf8).replace(dim_category_map).cast(pl.Int64).alias("category_key"),
            pl.struct(["pickup_zone", "pickup_date", "pickup_time"])
            .map_elements(
                lambda x: weather_lookup.get(
                    (
                        str(x["pickup_zone"]),
                        x["pickup_date"],
                        int(x["pickup_time"].hour),
                    ),
                ),
                return_dtype=pl.Int64,
            )
            .alias("weather_key"),
            (dropoff_dt - pickup_dt).dt.total_seconds().cast(pl.Int64).alias("trip_duration_seconds"),
        ],
    ).select(
        [
            "pickup_date_key",
            "dropoff_date_key",
            "pickup_time_key",
            "dropoff_time_key",
            "pickup_zone_key",
            "dropoff_zone_key",
            "vendor_key",
            "payment_type_key",
            "category_key",
            "weather_key",
            "passenger_count",
            "trip_distance",
            "tips",
            "base_fare",
            "tolls_and_fees",
            "taxes",
            "total_passenger_paid",
            "driver_payout",
            "driver_pay",
            "trip_duration_seconds",
        ],
    )


def reset_olap_schema() -> None:
    with olap_pipeline.sql_client() as client:
        client.execute_sql('DROP SCHEMA IF EXISTS "olap" CASCADE;')
        client.execute_sql('CREATE SCHEMA "olap";')


def main():
    source_schema = detect_latest_raw_schema()
    print(f"Using source schema: {source_schema}")

    dims = build_dimension_maps(source_schema)
    weather_df, weather_lookup = build_weather_dimension()

    min_date = dims["min_date"]
    max_date = dims["max_date"]
    holiday_set = dims["holiday_set"]

    dim_zone_map = dims["dim_zone_map"]
    dim_vendor_map = dims["dim_vendor_map"]
    dim_payment_map = dims["dim_payment_map"]
    dim_category_map = dims["dim_category_map"]

    dim_date_df = make_dim_date_table(min_date, max_date, holiday_set)
    dim_time_df = make_dim_time_table(dims["time_list"])
    dim_zone_df = make_dim_lookup_table(dims["zone_list"], "zone_key", "zone_name")
    dim_vendor_df = make_dim_lookup_table(dims["vendor_list"], "vendor_key", "vendor_name")
    dim_payment_df = make_dim_lookup_table(dims["payment_type_list"], "payment_type_key", "payment_type_name")
    dim_category_df = make_dim_lookup_table(dims["category_list"], "category_key", "category_name")

    @dlt.resource(name="dim_date", write_disposition="replace")
    def dim_date():
        yield dim_date_df.to_arrow()

    @dlt.resource(name="dim_time", write_disposition="replace")
    def dim_time():
        yield dim_time_df.to_arrow()

    @dlt.resource(name="dim_zone", write_disposition="replace")
    def dim_zone():
        yield dim_zone_df.to_arrow()

    @dlt.resource(name="dim_vendor", write_disposition="replace")
    def dim_vendor():
        yield dim_vendor_df.to_arrow()

    @dlt.resource(name="dim_payment_type", write_disposition="replace")
    def dim_payment_type():
        yield dim_payment_df.to_arrow()

    @dlt.resource(name="dim_category", write_disposition="replace")
    def dim_category():
        yield dim_category_df.to_arrow()

    @dlt.resource(name="dim_weather", write_disposition="replace")
    def dim_weather():
        yield weather_df.to_arrow()

    @dlt.resource(name="fact_trip", write_disposition="replace")
    def fact_trip():
        for source_df in iter_source_frames(source_schema):
            if source_df.is_empty():
                continue

            # Keep only the columns needed for the fact table.
            batch = source_df.select(
                [
                    "pickup_date",
                    "dropoff_date",
                    "pickup_time",
                    "dropoff_time",
                    "pickup_zone",
                    "dropoff_zone",
                    "vendor",
                    "payment_type",
                    "category",
                    "passenger_count",
                    "trip_distance",
                    "tips",
                    "base_fare",
                    "tolls_and_fees",
                    "taxes",
                    "total_passenger_paid",
                    "driver_payout",
                    "driver_pay",
                ],
            )

            batch = batch.filter(
                pl.col("pickup_date").is_not_null()
                & pl.col("dropoff_date").is_not_null()
                & pl.col("pickup_time").is_not_null()
                & pl.col("dropoff_time").is_not_null(),
            )

            if batch.is_empty():
                continue

            fact_df = make_fact_batch(
                batch,
                dim_zone_map=dim_zone_map,
                dim_vendor_map=dim_vendor_map,
                dim_payment_map=dim_payment_map,
                dim_category_map=dim_category_map,
                weather_lookup=weather_lookup,
            )

            yield fact_df.to_arrow()

    _ = olap_pipeline.drop()
    reset_olap_schema()

    load_info = olap_pipeline.run(
        [
            dim_date(),
            dim_time(),
            dim_zone(),
            dim_vendor(),
            dim_payment_type(),
            dim_category(),
            dim_weather(),
            fact_trip(),
        ],
    )

    print(load_info)


if __name__ == "__main__":
    main()
