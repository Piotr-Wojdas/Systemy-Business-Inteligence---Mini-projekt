from datetime import date

import niquests as requests
import polars as pl
from tqdm import tqdm

input_path = "data/lookup/taxi_zone_lookup.csv"
output_path = "data/thirdparty/weather.csv"

start_date = "2026-01-01"
end_date = "2026-01-31"

df_zones = pl.read_csv(input_path)

df_zones = df_zones.select(["Zone", "Borough", "LocationID"])

df_zones = df_zones.with_columns(
    pl.when(pl.col("LocationID") == 132)
    .then(pl.lit("JFK"))
    .when(pl.col("Borough") == "Queens")
    .then(pl.lit("LGA"))
    .otherwise(pl.lit("CentralPark"))
    .alias("weather_station"),
).drop(["Borough", "LocationID"])

stations = {
    "CentralPark": {"lat": 40.7829, "lon": -73.9654},
    "LGA": {"lat": 40.7772, "lon": -73.8726},
    "JFK": {"lat": 40.6398, "lon": -73.7789},
}

# Mapping WMO Weather Codes to descriptive strings
wmo_code_map = {
    0: "Sunny",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing Rime Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    56: "Light Freezing Drizzle",
    57: "Dense Freezing Drizzle",
    61: "Slight Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    66: "Light Freezing Rain",
    67: "Heavy Freezing Rain",
    71: "Slight Snowfall",
    73: "Moderate Snowfall",
    75: "Heavy Snowfall",
    77: "Snow Grains",
    80: "Slight Rain Showers",
    81: "Moderate Rain Showers",
    82: "Violent Rain Showers",
    85: "Slight Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm with Slight Hail",
    99: "Thunderstorm with Heavy Hail",
}

weather_frames = []
for station_name, coords in tqdm(stations.items(), desc="Fetching data..."):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,snowfall,weather_code",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }

    response = requests.get(url, params=params).json()
    hourly_data = response["hourly"]

    df_station_weather = pl.DataFrame(
        {
            "datetime": hourly_data["time"],  # Format returned: "2026-01-01T00:00"
            "weather_station": station_name,
            "temperature_f": hourly_data["temperature_2m"],
            "precipitation_inches": hourly_data["precipitation"],
            "snowfall_inches": hourly_data["snowfall"],
            "weather_code": hourly_data["weather_code"],
        },
    )
    weather_frames.append(df_station_weather)

df_all_weather = pl.concat(weather_frames)

df_all_weather = (
    df_all_weather.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%dT%H:%M"))
    .with_columns(
        [pl.col("datetime").dt.strftime("%Y-%m-%d").alias("date"), pl.col("datetime").dt.hour().alias("hour")],
    )
    .drop("datetime")
)

df_all_weather = df_all_weather.with_columns(
    pl.col("weather_code").replace(wmo_code_map, default="Unknown").alias("weather_status"),
).drop("weather_code")

df_calendar_base = pl.datetime_range(start=date(2026, 1, 1), end=date(2026, 1, 31), interval="1h", eager=True).to_frame(
    "datetime",
)

df_calendar_base = df_calendar_base.with_columns(
    [pl.col("datetime").dt.strftime("%Y-%m-%d").alias("date"), pl.col("datetime").dt.hour().alias("hour")],
).drop("datetime")

df_zone_calendar = df_zones.join(df_calendar_base, how="cross")

df_final_weather = df_zone_calendar.join(
    df_all_weather,
    on=["date", "hour", "weather_station"],
    how="left",
).drop("weather_station")

df_final_weather = df_final_weather.select(
    ["Zone", "date", "hour", "temperature_f", "precipitation_inches", "snowfall_inches", "weather_status"],
)

df_final_weather.write_csv(output_path)
print("Finished")
