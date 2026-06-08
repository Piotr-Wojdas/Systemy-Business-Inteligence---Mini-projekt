set dotenv-load

default: run

run:
    uv run pipeline.py

install:
    uv venv && uv sync

makeenv:
    cp ./.env.example ./.env

up:
    docker compose up --build

down:
    docker compose down -v

dlt-drop-pending:
    uv run dlt pipeline taxi_pipeline drop-pending-packages
    uv run dlt pipeline taxi_pipeline drop

restart: down dlt-drop-pending up run
