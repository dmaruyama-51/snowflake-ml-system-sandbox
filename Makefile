.PHONY: test lint format all deploy-prediction-sproc deploy-training-sproc deploy-sproc

POETRY = $(shell which poetry)
POETRY_OPTION = --no-interaction --no-ansi
POETRY_RUN = ${POETRY} run ${POETRY_OPTION}

MYPY_OPTIONS = --install-types --non-interactive --ignore-missing-imports

# ==============================
# dev
# ==============================

test: 
	${POETRY_RUN} pytest tests/
lint: 
	${POETRY_RUN} mypy ${MYPY_OPTIONS} -p src -p tests
	${POETRY_RUN} ruff check . --extend-select I --fix
format: 
	${POETRY_RUN} ruff format .

all: test lint format

# ==============================
# deploy sproc
# ==============================

deploy-sproc-prediction:
	${POETRY_RUN} python tests/pipelines/test_sproc_prediction.py

deploy-sproc-training:
	${POETRY_RUN} python tests/pipelines/test_sproc_training.py

deploy-sproc: deploy-prediction-sproc deploy-training-sproc 