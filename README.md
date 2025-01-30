# Snowflake ML System Sandbox

## Overview

This repository serves as a personal sandbox for exploring and staying up-to-date with Snowflake's latest machine learning features. The primary focus is on implementing practical use cases to deepen understanding and evaluate the integration of Snowflake’s capabilities with Python-based workflows. Specifically, this sandbox demonstrates predicting and scoring customer purchasing intent in an online shopping context, storing these scores in Snowflake for downstream marketing applications.

## Usecase

The system predicts customer purchasing intent based on session data from an online shopping platform. The daily batch processing pipeline:
- Computes purchasing intent scores for each user.
- Stores the scores in the Scores table in Snowflake.

This setup simulates a marketing pipeline, enabling targeted campaigns based on intent scores.

### Dataset
The dataset used is a modified version of the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) from the UCI Machine Learning Repository. Modifications include:
- `SessionDate`: Derived from the Month column and formatted as 2024-xx-01.
- `UID`: Generated using UUIDs.

By combining SessionDate and UID, each record in the dataset is uniquely identifiable. 

## Technical Stack

- Programming Language: Python 3.11
- Dependency Management: Poetry
- Continuous Integration: GitHub Actions
- Snowflake Features:
    - Snowpark for Python: A development framework to execute Python directly within Snowflake’s processing engine. 
    - Python Stored Procedures: Wrapping and deploying processing logic.
    - Model Registry: Managing machine learning models.
    - Task Scheduling: Automating daily score computations.
    - Feature Store: Managing reusable features for ML models (Planned).
    - Model Observability: Tracking model performance health (Planned).
    - Model Explainability: Support for calculating shaplay values (Planned).

## Development Environment

### Code Quality Management
- Ruff: Linting and formatting.
- Mypy: Static type checking.

### Development Commands

A Makefile is provided for streamlined development tasks:
- `make lint`: Run linter to check code quality.
- `make format`: Run formatter to ensure consistent code style.
- `make test`: Run tests using pytest.



## Setup

1. Set Up Python Environment:
```bash
poetry install
```

2. Configure Snowflake:
- Create a file named connection_parameters.json in the root directory with the following structure:
  ```json
  {
    "account": "",
    "user": "",
    "password": "",
    "role": "",
    "warehouse": "",
    "database": "",
    "schema": ""
  }
  ```
  - Fill in the necessary connection details specific to your Snowflake account.

3. Prepare and Upload Dataset
- Run the following command to preprocess the dataset and upload it to Snowflake:
  ```bash
  poetry run python src/adhoc/prepare_dataset.py
  ```
