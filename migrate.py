import os

import dlt
from dlt.sources.sql_database import sql_database
from dotenv import load_dotenv

load_dotenv()


def migrate_pg_to_mssql():
    pg_url = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@localhost:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
    )

    mssql_url = (
        f"mssql://sa:{os.getenv('MSSQL_SA_PASSWORD')}@localhost:{os.getenv('MSSQL_PORT', '1433')}/etl"  # CRETE DATABASE etl
        f"?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&Encrypt=yes"
    )

    postgres_source = sql_database(
        credentials=pg_url,
        schema="olap",
        table_names=[
            "dim_category",
            "dim_date",
            "dim_payment_type",
            "dim_time",
            "dim_vendor",
            "dim_weather",
            "dim_zone",
            "fact_trip",
        ],
    )

    pipeline = dlt.pipeline(
        pipeline_name="postgres_to_mssql",
        destination=dlt.destinations.mssql(mssql_url),
        dataset_name="olap",
        progress="enlighten",
    )

    print(pipeline.run(postgres_source, write_disposition="replace"))


if __name__ == "__main__":
    migrate_pg_to_mssql()
