docker exec -it etl-projekt-mssql-1 \
    /opt/mssql-tools18/bin/sqlcmd \
    -S localhost -U sa -P "PassworD123123" -C \
    -Q "CREATE DATABASE etl"
