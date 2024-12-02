from src.utils.snowflake import create_session
from src.data.loader import fetch_dataset
from src.data.preprocessing import split_data
from snowflake.snowpark import Session
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
    try:
        df = fetch_dataset(session)
        if df is None:
            raise ValueError("データセットが取得できませんでした")
        log_to_snowflake(session, f"データセットのフェッチ完了。行数: {len(df)}")

        df_train_val, df_test = split_data(df)
        log_to_snowflake(
            session,
            f"データセットの分割完了。学習検証データ: {len(df_train_val)}行, テストデータ: {len(df_test)}行",
        )

        return 1

    except Exception as e:
        log_to_snowflake(session, f"エラーが発生しました: {str(e)}")
        raise e


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
                (os.path.join(IMPORTS_DIR, "data"), "src.data"),
                (os.path.join(IMPORTS_DIR, "utils/config.py"), "src.utils.config"),
                os.path.join(IMPORTS_DIR, "config.yml"),
            ],
            "replace": True,
            "execute_as": "caller",
        }
        session.sproc.register(func=training_sproc, **sproc_config)  # type: ignore

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
