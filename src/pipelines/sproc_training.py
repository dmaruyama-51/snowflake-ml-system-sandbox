from src.utils.snowflake import create_session
from src.utils.logger import log_to_snowflake
from src.data.loader import fetch_dataset
from src.data.preprocessing import split_data
from src.models.trainer import train_model
from snowflake.snowpark import Session
from snowflake.ml.registry import Registry
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")

def sproc_training(session: Session) -> int:
    try:
        df = fetch_dataset(session, is_training=True)
        if df is None:
            raise ValueError("データセットが取得できませんでした")
        log_to_snowflake(session, f"データセットのフェッチ完了。行数: {len(df)}")

        df_train_val, df_test = split_data(df)
        log_to_snowflake(
            session,
            f"データセットの分割完了。学習検証データ: {len(df_train_val)}行, テストデータ: {len(df_test)}行",
        )

        model_pipeline, val_scores = train_model(df_train_val)
        log_to_snowflake(session, "モデルの学習完了")

        registry = Registry(session=session)
        _ = registry.log_model(
            model=model_pipeline,
            model_name="random_forest",
            version_name="v0_1_0",
            metrics=val_scores[0],
            sample_input_data=df_train_val.head(1),  # サンプル入力データを追加
        )
        log_to_snowflake(session, "モデルのログ完了")

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
                "snowflake-ml-python",
                "scikit-learn",
                "pandas",
                "numpy",
            ],
            "imports": [
                (os.path.join(IMPORTS_DIR, "data"), "src.data"),
                (os.path.join(IMPORTS_DIR, "models"), "src.models"),
                (os.path.join(IMPORTS_DIR, "utils/config.py"), "src.utils.config"),
                (os.path.join(IMPORTS_DIR, "utils/logger.py"), "src.utils.logger"),
                os.path.join(IMPORTS_DIR, "config.yml"),
            ],
            "replace": True,
            "execute_as": "caller",
        }
        session.sproc.register(func=sproc_training, **sproc_config)  # type: ignore

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
