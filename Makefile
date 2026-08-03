.PHONY: help install download-data infra-up infra-down kafka-topic \
        produce dbt-run dbt-test train serve test lint clean

PYTHON ?= python
VENV ?= .venv

help:
	@echo "Fraud Detection Feature Store"
	@echo ""
	@echo "  make install       Create venv and install requirements.txt"
	@echo "  make download-data Download creditcard.csv from Kaggle into data/raw/"
	@echo "  make infra-up      Start local Kafka + Redis via docker-compose"
	@echo "  make infra-down    Stop and remove local Kafka + Redis containers"
	@echo "  make kafka-topic   Create the Kafka topic with sane partitions/retention"
	@echo "  make produce       Run the Kafka producer against data/raw/creditcard.csv"
	@echo "  make dbt-run       Run dbt silver feature models"
	@echo "  make dbt-test      Run dbt tests (schema + data assertions)"
	@echo "  make train         Run model training (time-split + SMOTE + MLflow)"
	@echo "  make serve         Start the FastAPI real-time scoring service"
	@echo "  make test          Run the pytest suite"
	@echo "  make lint          Run ruff over the codebase"
	@echo "  make clean         Remove caches, venv, and local streaming/dbt state"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt

download-data:
	bash scripts/download_data.sh

infra-up:
	docker compose up -d
	@echo "Waiting for Kafka + Redis health checks..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' fraud-kafka 2>/dev/null)" = "healthy" ] && \
	       [ "$$(docker inspect -f '{{.State.Health.Status}}' fraud-redis 2>/dev/null)" = "healthy" ]; do \
	  sleep 2; \
	done
	@echo "Kafka + Redis are up."

infra-down:
	docker compose down

kafka-topic:
	bash scripts/create_kafka_topic.sh

produce:
	$(PYTHON) streaming/producer.py

dbt-run:
	cd dbt && dbt run

dbt-test:
	cd dbt && dbt test

train:
	$(PYTHON) notebooks/02_model_training.py

serve:
	uvicorn serving.scoring_api:app --host $${SCORING_API_HOST:-0.0.0.0} --port $${SCORING_API_PORT:-8080}

test:
	pytest tests/ -v

lint:
	ruff check .

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache dbt/target dbt/dbt_packages dbt/logs
	find . -type d -name "__pycache__" -exec rm -rf {} +
