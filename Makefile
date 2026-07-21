# Atalhos do dia a dia. `make help` lista tudo.
#
# No Windows use `.venv\Scripts\python.exe -m ...` diretamente, ou rode via
# Git Bash / WSL.

PYTHON := .venv/bin/python
ifeq ($(OS),Windows_NT)
	PYTHON := .venv/Scripts/python.exe
endif

.DEFAULT_GOAL := help
.PHONY: help setup test test-fast test-cov lint format typecheck check arch \
        serve report drift calibrate ingest sample docker-build docker-up \
        docker-down clean

help: ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Cria o venv e instala as dependências de desenvolvimento
	uv venv --python 3.11
	uv pip install --python $(PYTHON) -e ".[dev]"
	@echo "pronto. ative com: source .venv/bin/activate"

setup-ml: ## Instala também torch, xgboost e mlflow (extra [ml])
	uv pip install --python $(PYTHON) -e ".[dev,ml]"

# ── qualidade ────────────────────────────────────────────────────────────────
test: ## Roda a suíte completa
	$(PYTHON) -m pytest

test-fast: ## Só domínio e aplicação — deve rodar em menos de 1s
	$(PYTHON) -m pytest tests/unit -q

test-cov: ## Suíte com relatório de cobertura
	$(PYTHON) -m pytest --cov=src/eolica --cov-report=term-missing --cov-report=html

arch: ## Verifica as regras de dependência entre camadas
	$(PYTHON) -m pytest tests/architecture -v

lint: ## Ruff
	$(PYTHON) -m ruff check .

format: ## Formata com Ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck: ## Mypy estrito
	$(PYTHON) -m mypy

check: lint typecheck test ## Tudo que o CI roda, localmente

# ── operação ─────────────────────────────────────────────────────────────────
serve: ## Sobe a API em modo desenvolvimento
	$(PYTHON) -m eolica.interfaces.cli.main serve --reload

report: ## Relatório de um dia — make report DAY=2022-01-20
	$(PYTHON) -m eolica.interfaces.cli.main report $(DAY)

drift: ## Relatório de drift
	$(PYTHON) -m eolica.interfaces.cli.main drift

calibrate: ## Calibra o detector e mostra o limiar
	$(PYTHON) -m eolica.interfaces.cli.main calibrate

ingest: ## Reamostra o SCADA bruto para a grade de 10 minutos
	$(PYTHON) -m eolica.interfaces.cli.main ingest

sample: ## Regenera o recorte de dados versionado
	$(PYTHON) scripts/make_sample.py

# ── docker ───────────────────────────────────────────────────────────────────
docker-build: ## Builda a imagem da API
	docker build -f docker/api.Dockerfile -t eolica-scada:local .

docker-up: ## Sobe API + MLflow + Prometheus
	docker compose up -d --build

docker-down: ## Derruba a stack
	docker compose down

clean: ## Remove caches e artefatos de build
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml junit.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
