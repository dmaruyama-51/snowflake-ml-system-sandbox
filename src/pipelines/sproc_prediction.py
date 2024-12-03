import os
import sys
from snowflake.snowpark import Session
from src.utils.snowflake import create_session
from src.data.loader import fetch_dataset
from src.models.predictor import load_latest_model, predict


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")

def sproc_prediction(session: Session) -> int:
    """推論用のSprocを登録する関数"""
    try:
        df = fetch_dataset(session, is_training=False)
        if df is None:
            raise ValueError("データセットが取得できませんでした")

        model = load_latest_model(session)
        _ = predict(df, model)
        
        return 1

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        session = create_session()
        sproc_config = {
            "name": "PREDICTION",
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
                os.path.join(IMPORTS_DIR, "config.yml")
            ],
            "replace": True,
            "execute_as": "caller",
        }
        session.sproc.register(func=sproc_prediction, **sproc_config)  # type: ignore

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
