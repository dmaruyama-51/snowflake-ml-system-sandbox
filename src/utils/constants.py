import os
from src.utils.config import load_config

config = load_config()

# Snowflake関連の定数
DATABASE_DEV = config["data"]["snowflake"]["database_dev"]
SCHEMA = config["data"]["snowflake"]["schema"]
DATASET = config["data"]["snowflake"]["dataset_table"]
SOURCE = config["data"]["snowflake"]["source_table"]

CATEGORICAL_FEATURES = config["data"]["features"]["categorical"]
NUMERICAL_FEATURES = config["data"]["features"]["numeric"]
TARGET = config["data"]["target"]

# ディレクトリパス関連の定数
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")