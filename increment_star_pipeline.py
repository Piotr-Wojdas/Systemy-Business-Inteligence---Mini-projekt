import os
from collections.abc import Iterator
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

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")

PG_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"
SQLALCHEMY_URL = PG_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

os.environ["DESTINATION__POSTGRES__CREDENTIALS"] = PG_URL

WEATHER_CSV = Path("data/thirdparty/weather.csv")
LOG_DIR = Path("logs")
SUMMARY_LOG_FILE = LOG_DIR / "olap_summary.log"
VALIDATION_LOG_FILE = LOG_DIR / "olap_validation.log"

olap_pipeline = dlt.pipeline(
    pipeline_name="taxi_star_schema_increment",
    destination="postgres",
    dataset_name="olap",
    progress="enlighten",
)

FACT_COLUMNS = [
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
]

# Exact-row dedupe key for merge loading.
# Compound keys are allowed by dlt.
FACT_PRIMARY_KEY = FACT_COLUMNS.copy()


def append_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


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


def read_olap_table_df(table_name: str) -> pl.DataFrame:
    try:
        batches: list[pl.DataFrame] = []
        source = sql_table(
            credentials=PG_URL,
            schema="olap",
            table=table_name,
            backend="pyarrow",
            chunk_size=20000,
            reflection_level="full",
        )
        for batch in source:
            batches.append(pl.from_arrow(batch))
        return pl.concat(batches, how="vertical") if batches else pl.DataFrame()
    except Exception:  # noqa: BLE001
        return pl.DataFrame()


def raw_source_relation(source_schema: str):
    return sql_table(
        credentials=PG_URL,
        schema=source_schema,
        table="imported",
        backend="pyarrow",
        chunk_size=20000,
        reflection_level="full",
    )


def iter_source_frames(source_schema: str) -> Iterator[pl.DataFrame]:
    source = raw_source_relation(source_schema)
    for batch in source:
        yield pl.from_arrow(batch)


def count_nulls(df: pl.DataFrame, column: str) -> int:
    return df.select(pl.col(column).is_null().sum()).item()


def collect_source_uniques(source_schema: str):
    min_date = None
    max_date = None

    unique_times = set()
    unique_zones = set()
    unique_vendors = set()
    unique_payment_types = set()
    unique_categories = set()

    total_rows = 0
    missing_pickup_date_rows = 0
    missing_dropoff_date_rows = 0
    missing_pickup_time_rows = 0
    missing_dropoff_time_rows = 0
    invalid_date_order_rows = 0
    missing_pickup_zone_rows = 0
    missing_dropoff_zone_rows = 0
    missing_vendor_rows = 0
    missing_payment_rows = 0
    missing_category_rows = 0

    for df in iter_source_frames(source_schema):
        if df.is_empty():
            continue

        total_rows += df.height

        missing_pickup_date_rows += count_nulls(df, "pickup_date")
        missing_dropoff_date_rows += count_nulls(df, "dropoff_date")
        missing_pickup_time_rows += count_nulls(df, "pickup_time")
        missing_dropoff_time_rows += count_nulls(df, "dropoff_time")

        if "pickup_zone" in df.columns:
            missing_pickup_zone_rows += count_nulls(df, "pickup_zone")
        if "dropoff_zone" in df.columns:
            missing_dropoff_zone_rows += count_nulls(df, "dropoff_zone")
        if "vendor" in df.columns:
            missing_vendor_rows += count_nulls(df, "vendor")
        if "payment_type" in df.columns:
            missing_payment_rows += count_nulls(df, "payment_type")
        if "category" in df.columns:
            missing_category_rows += count_nulls(df, "category")

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

        invalid_date_order_rows += df.select((pl.col("dropoff_date") < pl.col("pickup_date")).sum()).item()

    if min_date is None or max_date is None:
        msg = "Source table is empty or date columns are missing."
        raise RuntimeError(msg)

    append_log_line(
        VALIDATION_LOG_FILE,
        (
            f"{source_schema};rows={total_rows};"
            f"missing_pickup_date={missing_pickup_date_rows};"
            f"missing_dropoff_date={missing_dropoff_date_rows};"
            f"missing_pickup_time={missing_pickup_time_rows};"
            f"missing_dropoff_time={missing_dropoff_time_rows};"
            f"invalid_date_order={invalid_date_order_rows};"
            f"missing_pickup_zone={missing_pickup_zone_rows};"
            f"missing_dropoff_zone={missing_dropoff_zone_rows};"
            f"missing_vendor={missing_vendor_rows};"
            f"missing_payment={missing_payment_rows};"
            f"missing_category={missing_category_rows}"
        ),
    )

    assert missing_pickup_date_rows == 0, f"{source_schema}: missing pickup_date rows = {missing_pickup_date_rows}"
    assert missing_dropoff_date_rows == 0, f"{source_schema}: missing dropoff_date rows = {missing_dropoff_date_rows}"
    assert missing_pickup_time_rows == 0, f"{source_schema}: missing pickup_time rows = {missing_pickup_time_rows}"
    assert missing_dropoff_time_rows == 0, f"{source_schema}: missing dropoff_time rows = {missing_dropoff_time_rows}"
    assert invalid_date_order_rows == 0, f"{source_schema}: invalid date order rows = {invalid_date_order_rows}"

    zone_list = sorted(unique_zones)
    vendor_list = sorted(unique_vendors)
    payment_type_list = sorted(unique_payment_types)
    category_list = sorted(unique_categories)
    time_list = sorted(unique_times, key=lambda t: (t.hour, t.minute, t.second))

    return {
        "min_date": min_date,
        "max_date": max_date,
        "zone_list": zone_list,
        "vendor_list": vendor_list,
        "payment_type_list": payment_type_list,
        "category_list": category_list,
        "time_list": time_list,
        "holiday_set": holidays.US(years=range(min_date.year, max_date.year + 1)),
    }


def build_static_lookup_state(
    existing_df: pl.DataFrame,
    key_col: str,
    value_col: str,
) -> tuple[dict[str, int], int]:
    mapping: dict[str, int] = {}
    max_key = 0

    if existing_df.is_empty():
        return mapping, max_key

    for row in existing_df.select([key_col, value_col]).iter_rows(named=True):
        if row[key_col] is None or row[value_col] is None:
            continue
        key = int(row[key_col])
        value = str(row[value_col])
        mapping[value] = key
        max_key = max(max_key, key)

    return mapping, max_key


def build_time_lookup_state(existing_df: pl.DataFrame) -> set[time]:
    if existing_df.is_empty():
        return set()
    return set(existing_df.get_column("full_time").drop_nulls().to_list())


def build_date_lookup_state(existing_df: pl.DataFrame) -> set[date]:
    if existing_df.is_empty():
        return set()
    return set(existing_df.get_column("full_date").drop_nulls().to_list())


def build_weather_lookup_state(existing_df: pl.DataFrame) -> tuple[dict[tuple[str, date, int], int], int]:
    mapping: dict[tuple[str, date, int], int] = {}
    max_key = 0

    if existing_df.is_empty():
        return mapping, max_key

    for row in existing_df.select(
        ["weather_key", "zone_name", "weather_date", "weather_hour"],
    ).iter_rows(named=True):
        if row["weather_key"] is None:
            continue
        key = (
            str(row["zone_name"]),
            norm_date(row["weather_date"]),
            int(row["weather_hour"]),
        )
        if key[1] is None:
            continue
        mapping[key] = int(row["weather_key"])
        max_key = max(max_key, int(row["weather_key"]))

    return mapping, max_key


def build_incremental_lookup_df(
    source_values: set[str],
    existing_map: dict[str, int],
    key_name: str,
    value_name: str,
) -> tuple[pl.DataFrame, dict[str, int]]:
    missing = sorted(v for v in source_values if v not in existing_map)
    next_key = max(existing_map.values(), default=0)

    rows = []
    updated_map = dict(existing_map)
    for idx, value in enumerate(missing, start=1):
        key = next_key + idx
        rows.append({key_name: key, value_name: value})
        updated_map[value] = key

    return pl.DataFrame(rows), updated_map


def build_dim_date_rows(
    min_date: date, max_date: date, existing_dates: set[date], holiday_set: set[date]
) -> pl.DataFrame:
    rows = []
    cur = min_date
    while cur <= max_date:
        if cur not in existing_dates:
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


def build_dim_time_rows(time_list: list[time], existing_times: set[time]) -> pl.DataFrame:
    missing_times = [
        t for t in sorted(time_list, key=lambda t: (t.hour, t.minute, t.second)) if t not in existing_times
    ]
    return pl.DataFrame(
        {
            "time_key": [time_key(t) for t in missing_times],
            "full_time": missing_times,
            "hour": [t.hour for t in missing_times],
            "minute": [t.minute for t in missing_times],
            "second": [t.second for t in missing_times],
            "part_of_day": [part_of_day(t) for t in missing_times],
        },
    )


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
        .select(
            [
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

    return weather_df


def build_incremental_weather_rows(
    weather_df: pl.DataFrame,
    existing_weather_map: dict[tuple[str, date, int], int],
) -> tuple[pl.DataFrame, dict[tuple[str, date, int], int]]:
    next_key = max(existing_weather_map.values(), default=0)
    rows = []
    updated_map = dict(existing_weather_map)

    for row in weather_df.iter_rows(named=True):
        nat_key = (str(row["zone_name"]), row["weather_date"], int(row["weather_hour"]))
        if nat_key in updated_map:
            continue
        next_key += 1
        updated_map[nat_key] = next_key
        rows.append(
            {
                "weather_key": next_key,
                "zone_name": row["zone_name"],
                "weather_date": row["weather_date"],
                "weather_hour": int(row["weather_hour"]),
                "temperature_f": row["temperature_f"],
                "temperature_c": row["temperature_c"],
                "precipitation_inches": row["precipitation_inches"],
                "snowfall_inches": row["snowfall_inches"],
                "weather_status": row["weather_status"],
            },
        )

    return pl.DataFrame(rows), updated_map


def validate_cost_columns(df: pl.DataFrame, category: str) -> None:
    tol = 1e-6

    if category in ("yellow", "green"):
        base_fare_mismatch = df.select(
            (
                (pl.col("base_fare") - (pl.col("fare_amount").fill_null(0) + pl.col("extra").fill_null(0))).abs() > tol
            ).sum(),
        ).item()

        if category == "yellow":
            tolls_mismatch = df.select(
                (
                    (
                        pl.col("tolls_and_fees")
                        - (
                            pl.col("tolls_amount").fill_null(0)
                            + pl.col("congestion_surcharge").fill_null(0)
                            + pl.col("cbd_congestion_fee").fill_null(0)
                            + pl.col("Airport_fee").fill_null(0)
                        )
                    ).abs()
                    > tol
                ).sum(),
            ).item()
        else:
            tolls_mismatch = df.select(
                (
                    (
                        pl.col("tolls_and_fees")
                        - (
                            pl.col("tolls_amount").fill_null(0)
                            + pl.col("congestion_surcharge").fill_null(0)
                            + pl.col("cbd_congestion_fee").fill_null(0)
                        )
                    ).abs()
                    > tol
                ).sum(),
            ).item()

        taxes_mismatch = df.select(
            (
                (
                    pl.col("taxes") - (pl.col("mta_tax").fill_null(0) + pl.col("improvement_surcharge").fill_null(0))
                ).abs()
                > tol
            ).sum(),
        ).item()

        total_paid_mismatch = df.select(
            ((pl.col("total_passenger_paid") - pl.col("total_amount")).abs() > tol).sum(),
        ).item()

        driver_payout_mismatch = df.select(
            (
                (
                    pl.col("driver_payout")
                    - (
                        pl.col("fare_amount").fill_null(0)
                        + pl.col("extra").fill_null(0)
                        + pl.col("tolls_amount").fill_null(0)
                        + pl.col("tip_amount").fill_null(0)
                    )
                ).abs()
                > tol
            ).sum(),
        ).item()

        tips_rule_mismatch = df.select(
            (
                (
                    (pl.col("payment_type") == "Credit card")
                    & pl.col("tip_amount").is_not_null()
                    & ((pl.col("tips") - pl.col("tip_amount")).abs() > tol)
                )
                | ((pl.col("payment_type") != "Credit card") & pl.col("tips").is_not_null())
            ).sum(),
        ).item()

        append_log_line(
            VALIDATION_LOG_FILE,
            (
                f"costs;category={category};"
                f"base_fare_mismatch={base_fare_mismatch};"
                f"tolls_and_fees_mismatch={tolls_mismatch};"
                f"taxes_mismatch={taxes_mismatch};"
                f"total_passenger_paid_mismatch={total_paid_mismatch};"
                f"driver_payout_mismatch={driver_payout_mismatch};"
                f"tips_rule_mismatch={tips_rule_mismatch}"
            ),
        )

        assert base_fare_mismatch == 0, f"{category}: base_fare mismatch rows = {base_fare_mismatch}"
        assert tolls_mismatch == 0, f"{category}: tolls_and_fees mismatch rows = {tolls_mismatch}"
        assert taxes_mismatch == 0, f"{category}: taxes mismatch rows = {taxes_mismatch}"
        assert total_paid_mismatch == 0, f"{category}: total_passenger_paid mismatch rows = {total_paid_mismatch}"
        assert driver_payout_mismatch == 0, f"{category}: driver_payout mismatch rows = {driver_payout_mismatch}"
        assert tips_rule_mismatch == 0, f"{category}: tips rule mismatch rows = {tips_rule_mismatch}"

    if category == "fhvhv":
        base_fare_mismatch = df.select(
            ((pl.col("base_fare") - pl.col("base_passenger_fare").fill_null(0)).abs() > tol).sum(),
        ).item()

        tolls_mismatch = df.select(
            (
                (
                    pl.col("tolls_and_fees")
                    - (
                        pl.col("tolls").fill_null(0)
                        + pl.col("congestion_surcharge").fill_null(0)
                        + pl.col("cbd_congestion_fee").fill_null(0)
                        + pl.col("airport_fee").fill_null(0)
                    )
                ).abs()
                > tol
            ).sum(),
        ).item()

        taxes_mismatch = df.select(
            ((pl.col("taxes") - (pl.col("sales_tax").fill_null(0) + pl.col("bcf").fill_null(0))).abs() > tol).sum(),
        ).item()

        total_paid_mismatch = df.select(
            (
                (
                    pl.col("total_passenger_paid")
                    - (
                        pl.col("base_passenger_fare").fill_null(0)
                        + pl.col("tolls").fill_null(0)
                        + pl.col("congestion_surcharge").fill_null(0)
                        + pl.col("cbd_congestion_fee").fill_null(0)
                        + pl.col("airport_fee").fill_null(0)
                        + pl.col("sales_tax").fill_null(0)
                        + pl.col("bcf").fill_null(0)
                        + pl.col("tips").fill_null(0)
                    )
                ).abs()
                > tol
            ).sum(),
        ).item()

        driver_payout_mismatch = df.select(
            (
                (
                    pl.col("driver_payout")
                    - (pl.col("driver_pay").fill_null(0) + pl.col("tolls").fill_null(0) + pl.col("tips").fill_null(0))
                ).abs()
                > tol
            ).sum(),
        ).item()

        append_log_line(
            VALIDATION_LOG_FILE,
            (
                f"costs;category={category};"
                f"base_fare_mismatch={base_fare_mismatch};"
                f"tolls_and_fees_mismatch={tolls_mismatch};"
                f"taxes_mismatch={taxes_mismatch};"
                f"total_passenger_paid_mismatch={total_paid_mismatch};"
                f"driver_payout_mismatch={driver_payout_mismatch}"
            ),
        )

        assert base_fare_mismatch == 0, f"{category}: base_fare mismatch rows = {base_fare_mismatch}"
        assert tolls_mismatch == 0, f"{category}: tolls_and_fees mismatch rows = {tolls_mismatch}"
        assert taxes_mismatch == 0, f"{category}: taxes mismatch rows = {taxes_mismatch}"
        assert total_paid_mismatch == 0, f"{category}: total_passenger_paid mismatch rows = {total_paid_mismatch}"
        assert driver_payout_mismatch == 0, f"{category}: driver_payout mismatch rows = {driver_payout_mismatch}"


def build_fact_batch(  # noqa: PLR0913
    df: pl.DataFrame,
    category: str,
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

    fact_df = df.with_columns(
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
    )

    validate_cost_columns(fact_df, category)

    return fact_df.select(FACT_COLUMNS)


def drop_schema(schema_name: str) -> None:
    engine = create_engine(SQLALCHEMY_URL)
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE;'))


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    source_schema = detect_latest_raw_schema()
    print(f"Using source schema: {source_schema}")

    source_uniques = collect_source_uniques(source_schema)
    weather_df = build_weather_dimension()

    existing_zone_df = read_olap_table_df("dim_zone")
    existing_vendor_df = read_olap_table_df("dim_vendor")
    existing_payment_df = read_olap_table_df("dim_payment_type")
    existing_category_df = read_olap_table_df("dim_category")
    existing_date_df = read_olap_table_df("dim_date")
    existing_time_df = read_olap_table_df("dim_time")
    existing_weather_df = read_olap_table_df("dim_weather")

    existing_zone_map, _ = build_static_lookup_state(existing_zone_df, "zone_key", "zone_name")
    existing_vendor_map, _ = build_static_lookup_state(existing_vendor_df, "vendor_key", "vendor_name")
    existing_payment_map, _ = build_static_lookup_state(existing_payment_df, "payment_type_key", "payment_type_name")
    existing_category_map, _ = build_static_lookup_state(existing_category_df, "category_key", "category_name")

    existing_dates = build_date_lookup_state(existing_date_df)
    existing_times = build_time_lookup_state(existing_time_df)
    existing_weather_map, _ = build_weather_lookup_state(existing_weather_df)

    dim_zone_df, dim_zone_map = build_incremental_lookup_df(
        source_uniques["unique_zones"],
        existing_zone_map,
        "zone_key",
        "zone_name",
    )
    dim_vendor_df, dim_vendor_map = build_incremental_lookup_df(
        source_uniques["unique_vendors"],
        existing_vendor_map,
        "vendor_key",
        "vendor_name",
    )
    dim_payment_df, dim_payment_map = build_incremental_lookup_df(
        source_uniques["unique_payment_types"],
        existing_payment_map,
        "payment_type_key",
        "payment_type_name",
    )
    dim_category_df, dim_category_map = build_incremental_lookup_df(
        source_uniques["unique_categories"],
        existing_category_map,
        "category_key",
        "category_name",
    )

    dim_date_df = build_dim_date_rows(
        source_uniques["min_date"],
        source_uniques["max_date"],
        existing_dates,
        source_uniques["holiday_set"],
    )
    dim_time_df = build_dim_time_rows(
        sorted(source_uniques["unique_times"], key=lambda t: (t.hour, t.minute, t.second)),
        existing_times,
    )
    dim_weather_df, weather_lookup = build_incremental_weather_rows(
        weather_df,
        existing_weather_map,
    )

    append_log_line(
        SUMMARY_LOG_FILE,
        (
            f"{source_schema};"
            f"min_date={source_uniques['min_date']};max_date={source_uniques['max_date']};"
            f"new_dim_date_rows={dim_date_df.height};"
            f"new_dim_time_rows={dim_time_df.height};"
            f"new_dim_zone_rows={dim_zone_df.height};"
            f"new_dim_vendor_rows={dim_vendor_df.height};"
            f"new_dim_payment_rows={dim_payment_df.height};"
            f"new_dim_category_rows={dim_category_df.height};"
            f"new_dim_weather_rows={dim_weather_df.height}"
        ),
    )

    @dlt.resource(name="dim_date", write_disposition="append")
    def dim_date():
        if dim_date_df.is_empty():
            return
        yield dim_date_df.to_arrow()

    @dlt.resource(name="dim_time", write_disposition="append")
    def dim_time():
        if dim_time_df.is_empty():
            return
        yield dim_time_df.to_arrow()

    @dlt.resource(name="dim_zone", write_disposition="append")
    def dim_zone():
        if dim_zone_df.is_empty():
            return
        yield dim_zone_df.to_arrow()

    @dlt.resource(name="dim_vendor", write_disposition="append")
    def dim_vendor():
        if dim_vendor_df.is_empty():
            return
        yield dim_vendor_df.to_arrow()

    @dlt.resource(name="dim_payment_type", write_disposition="append")
    def dim_payment_type():
        if dim_payment_df.is_empty():
            return
        yield dim_payment_df.to_arrow()

    @dlt.resource(name="dim_category", write_disposition="append")
    def dim_category():
        if dim_category_df.is_empty():
            return
        yield dim_category_df.to_arrow()

    @dlt.resource(name="dim_weather", write_disposition="append")
    def dim_weather():
        if dim_weather_df.is_empty():
            return
        yield dim_weather_df.to_arrow()

    @dlt.resource(
        name="fact_trip",
        write_disposition="merge",
        primary_key=FACT_PRIMARY_KEY,
    )
    def fact_trip():
        total_source_rows = 0
        total_after_required_filter = 0
        total_fact_rows = 0

        for source_df in iter_source_frames(source_schema):
            if source_df.is_empty():
                continue

            total_source_rows += source_df.height

            categories = sorted(
                {str(v) for v in source_df.get_column("category").drop_nulls().to_list() if v is not None},
            )

            for category in categories:
                category_batch = source_df.filter(pl.col("category") == category)

                if category_batch.is_empty():
                    continue

                selected_columns = [
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
                    "fare_amount",
                    "extra",
                    "tolls_amount",
                    "tip_amount",
                    "mta_tax",
                    "improvement_surcharge",
                    "congestion_surcharge",
                    "cbd_congestion_fee",
                    "Airport_fee",
                    "base_passenger_fare",
                    "tolls",
                    "sales_tax",
                    "bcf",
                    "airport_fee",
                ]

                batch = category_batch.select([c for c in selected_columns if c in category_batch.columns])

                required_before = batch.height

                batch = batch.filter(
                    pl.col("pickup_date").is_not_null()
                    & pl.col("dropoff_date").is_not_null()
                    & pl.col("pickup_time").is_not_null()
                    & pl.col("dropoff_time").is_not_null(),
                )

                removed_due_to_missing_datetime = required_before - batch.height
                total_after_required_filter += batch.height

                append_log_line(
                    VALIDATION_LOG_FILE,
                    (
                        f"{source_schema};category={category};batch_rows={required_before};"
                        f"removed_missing_datetime={removed_due_to_missing_datetime};"
                        f"kept_after_datetime_filter={batch.height}"
                    ),
                )

                assert removed_due_to_missing_datetime == 0, (
                    f"{source_schema}: removed rows due to missing date/time in fact stage = "
                    f"{removed_due_to_missing_datetime}"
                )

                if batch.is_empty():
                    continue

                fact_df = build_fact_batch(
                    batch,
                    category=category,
                    dim_zone_map=dim_zone_map,
                    dim_vendor_map=dim_vendor_map,
                    dim_payment_map=dim_payment_map,
                    dim_category_map=dim_category_map,
                    weather_lookup=weather_lookup,
                )

                mandatory_columns = [
                    "pickup_date_key",
                    "dropoff_date_key",
                    "pickup_time_key",
                    "dropoff_time_key",
                    "pickup_zone_key",
                    "dropoff_zone_key",
                    "vendor_key",
                    "payment_type_key",
                    "category_key",
                    "trip_duration_seconds",
                ]
                null_checks = {
                    col: fact_df.select(pl.col(col).is_null().sum()).item()
                    for col in mandatory_columns
                    if col in fact_df.columns
                }

                append_log_line(
                    VALIDATION_LOG_FILE,
                    f"{source_schema};category={category};fact_null_checks={null_checks}",
                )

                for col_name, null_count in null_checks.items():
                    assert null_count == 0, f"{source_schema}: nulls in mandatory fact column {col_name} = {null_count}"

                negative_duration_rows = fact_df.select((pl.col("trip_duration_seconds") < 0).sum()).item()
                append_log_line(
                    VALIDATION_LOG_FILE,
                    f"{source_schema};category={category};negative_trip_duration_rows={negative_duration_rows}",
                )
                assert negative_duration_rows == 0, (
                    f"{source_schema}: negative trip duration rows = {negative_duration_rows}"
                )

                assert fact_df.height == batch.height, (
                    f"{source_schema}: fact row count changed after reshape ({fact_df.height} != {batch.height})"
                )

                total_fact_rows += fact_df.height

                yield fact_df.to_arrow()

        append_log_line(
            SUMMARY_LOG_FILE,
            (
                f"{source_schema};fact_source_rows={total_source_rows};"
                f"fact_rows_after_datetime_filter={total_after_required_filter};"
                f"fact_rows_written={total_fact_rows}"
            ),
        )

    # remove dropping olap schema
    # _ = olap_pipeline.drop()
    # reset_olap_schema()

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

    drop_schema(source_schema)
    print(load_info)


if __name__ == "__main__":
    main()
