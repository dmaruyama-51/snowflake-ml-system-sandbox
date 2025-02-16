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
# streamlit
# ==============================

run-streamlit: __require_streamlit_app_name__
	${POETRY_RUN} streamlit run src/streamlit/${APP_NAME}/app.py

# ==============================
# deploy
# ==============================

deploy-sproc-dataset:
	${POETRY_RUN} python src/pipelines/sproc_dataset.py

deploy-sproc-prediction:
	${POETRY_RUN} python src/pipelines/sproc_prediction.py

deploy-sproc-training:
	${POETRY_RUN} python src/pipelines/sproc_training.py

deploy-sproc-offline-testing:
	${POETRY_RUN} python src/pipelines/sproc_offline_testing.py

deploy-sproc: deploy-sproc-dataset deploy-sproc-prediction deploy-sproc-training deploy-sproc-offline-testing

deploy-task-dataset:
	${POETRY_RUN} python src/tasks/task_dataset.py

deploy-task-prediction:
	${POETRY_RUN} python src/tasks/task_prediction.py

deploy-task-training:
	${POETRY_RUN} python src/tasks/task_training.py

deploy-task-offline-testing:
	${POETRY_RUN} python src/tasks/task_offline_testing.py

deploy-task: deploy-task-dataset deploy-task-prediction deploy-task-training deploy-task-offline-testing	

deploy-streamlit: __require_streamlit_app_name__
	cd src/streamlit/${APP_NAME} && \
		${POETRY_RUN} snow --config-file .snowflake/config.toml \
			streamlit deploy --replace

__require_streamlit_app_name__:
	@[ -n "$(APP_NAME)" ] || (echo "[ERROR] Parameter [APP_NAME] is requierd" 1>&2 && echo "(e.g) make xxx APP_NAME=hoge" 1>&2 && exit 1)