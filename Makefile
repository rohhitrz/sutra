.PHONY: help up down dev ingest eval test lint fmt migrate revision install

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## create the venv and install all deps (incl. dev)
	uv sync --extra dev

up:  ## start Postgres (+ Langfuse with: make up PROFILE=observability)
	docker compose $(if $(PROFILE),--profile $(PROFILE),) up -d

down:  ## stop the stack (all profiles)
	docker compose --profile observability down

dev:  ## run the API locally with reload
	uv run uvicorn sutra.api.app:app --reload --host 0.0.0.0 --port 8000

migrate:  ## apply all Alembic migrations
	uv run alembic upgrade head

revision:  ## create a new migration: make revision M="add foo"
	uv run alembic revision --autogenerate -m "$(M)"

ingest:  ## ingest data/corpus into Postgres
	uv run python -m sutra.ingestion.run

eval:  ## run the eval suite, write a report to evals/reports/
	uv run python -m evals.runners.run

test:  ## run the test suite
	uv run pytest

lint:  ## ruff (lint + format check) and pyright
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run pyright

fmt:  ## apply ruff formatting and autofixes
	uv run ruff check --fix src tests
	uv run ruff format src tests
