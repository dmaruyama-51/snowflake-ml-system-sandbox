.PHONY: setup test lint format all deploy-prediction-sproc deploy-training-sproc deploy-sproc

POETRY = $(shell which poetry)
POETRY_OPTION = --no-interaction --no-ansi
POETRY_RUN = ${POETRY} run ${POETRY_OPTION}

MYPY_OPTIONS = --install-types --non-interactive --ignore-missing-imports

# ==============================
# setup
# ==============================

setup:
	${POETRY_RUN} python src/setup.py

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
# deploy
# ==============================

deploy-sproc-dataset:
	${POETRY_RUN} python src/pipelines/sproc_dataset.py

deploy-sproc-prediction:
	${POETRY_RUN} python src/pipelines/sproc_prediction.py

deploy-sproc-training:
	${POETRY_RUN} python src/pipelines/sproc_training.py

deploy-sproc: deploy-sproc-dataset deploy-sproc-prediction deploy-sproc-training

deploy-task-dataset:
	${POETRY_RUN} python src/tasks/task_dataset.py

deploy-task-prediction:
	${POETRY_RUN} python src/tasks/task_prediction.py

deploy-task-training:
	${POETRY_RUN} python src/tasks/task_training.py

deploy-task: deploy-task-dataset deploy-task-prediction deploy-task-training