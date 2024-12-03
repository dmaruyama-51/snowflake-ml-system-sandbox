import os
import sys
from snowflake.snowpark import Session
from src.utils.snowflake import create_session
from src.utils.logger import log_to_snowflake
from src.data.loader import fetch_dataset
from src.models.predictor import load_latest_model, predict


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")

def sproc_prediction(session: Session) -> int:
    """
    推論Sprocの処理内容
    
    Args:
        session (Session): Snowflakeセッション
        
    Returns:
        int: 成功時は1、失敗時は例外を発生
        
    Raises:
        Exception: 処理中にエラーが発生した場合
    """
    log_to_snowflake(session, "推論処理を開始")
    try:
        df = fetch_dataset(session, is_training=False)
        if df is None:
            raise ValueError("データセットが取得できませんでした")
        log_to_snowflake(session, f"データセットのフェッチ完了。行数: {len(df)}")

        model = load_latest_model(session)
        log_to_snowflake(session, "モデルの読み込み完了")
        _ = predict(df, model)
        log_to_snowflake(session, "推論完了")

        # ToDo: 推論結果をスコアテーブルに書き込み
        
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
                (os.path.join(IMPORTS_DIR, "utils/logger.py"), "src.utils.logger"),
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
