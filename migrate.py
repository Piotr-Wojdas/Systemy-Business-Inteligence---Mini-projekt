import os

import dlt
from dlt.sources.sql_database import sql_database
from dotenv import load_dotenv

load_dotenv()


def migrate_pg_to_mssql():
    pg_user = os.getenv("POSTGRES_USER")
    pg_password = os.getenv("POSTGRES_PASSWORD")
    pg_db = os.getenv("POSTGRES_DB")
    pg_port = os.getenv("POSTGRES_PORT", "5432")

    pg_url = f"postgresql://{pg_user}:{pg_password}@localhost:{pg_port}/{pg_db}"

    mssql_url = (
        f"mssql+pyodbc://{ms_user}:{ms_password}@{ms_host}:1433/{ms_db}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=no"
    )

    postgres_source = sql_database(
        credentials=pg_url,
        schema="olap",
    )

    pipeline = dlt.pipeline(
        pipeline_name="postgres_to_mssql",
        destination="mssql",
        credentials=mssql_url,
        dataset_name="olap",
    )

    load_info = pipeline.run(postgres_source, write_disposition="replace")
    print(load_info)


if __name__ == "__main__":
    migrate_pg_to_mssql()
