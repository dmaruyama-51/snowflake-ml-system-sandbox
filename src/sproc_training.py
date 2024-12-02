from src.dwh.snowflake import create_session
from snowflake.snowpark import Session
from src.data.loader import fetch_dataset
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")

def log_to_snowflake(session: Session, message: str) -> None:
    session.sql(f"""
    INSERT INTO log_trace (
        timestamp,
        event_name,
        message
    ) VALUES (
        CURRENT_TIMESTAMP(),
        'TRAINING_LOG',
        '{message}'
    )
    """).collect()


def training_sproc(session: Session) -> int: 

    df = fetch_dataset(session)
    log_to_snowflake(session, f"データセットのフェッチ完了。行数: {len(df)}")
    print(df.head())

    return 1


if __name__ == "__main__":
    try:
        session = create_session()
        sproc_config = {
            "name": "TRAINING",
            "is_permanent": True,
            "stage_location": "@practice.ml.sproc",
            "packages": [
                "snowflake-snowpark-python",
                "scikit-learn",
                "pandas",
                "numpy",
            ],
            "imports": [
                (os.path.join(IMPORTS_DIR, "data"), "src.data")
            ],
            "replace": True,
            "execute_as": "caller",
        }
        session.sproc.register(func=training_sproc, **sproc_config)

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()