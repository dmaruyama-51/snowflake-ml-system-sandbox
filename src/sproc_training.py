from datetime import datetime
import sys
import os
from typing import Dict

import pandas as pd

# from src.data.loader import fetch_dataset
# from src.data.preprocessing import split_data
# from src.models.trainer import train_model
# from src.evaluation.metrics import calc_evaluation_metrics
from src.utils.snowflake import create_session #, upload_dataframe_to_snowflake
from snowflake.snowpark.session import Session
# ファイルパスの設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")

def training_sproc(session: Session) -> Dict:
    """モデルの学習を実行するストアドプロシージャ"""
    
    from src.utils.snowflake import upload_dataframe_to_snowflake
    from src.data.loader import fetch_dataset
    from src.data.preprocessing import split_data
    from src.models.trainer import train_model
    from src.evaluation.metrics import calc_evaluation_metrics

    try:
        df = fetch_dataset(session=session)
        df_train_val, df_test = split_data(df)
        execution_date = datetime.now().strftime("%Y%m%d%H%M%S")

        upload_dataframe_to_snowflake(
            session=session,
            df=df_train_val,
            database_name="ML_DB",
            schema_name="TRAINING",
            table_name=f"DATASET_TRAIN_{execution_date}"
        )
        upload_dataframe_to_snowflake(
            session=session,
            df=df_test,
            database_name="ML_DB",
            schema_name="TRAINING",
            table_name=f"DATASET_TEST_{execution_date}"
        )

        model = train_model(df_train_val)
        scores = calc_evaluation_metrics(model, df_test)
        
        return scores

    except Exception as e:
        raise RuntimeError(f"モデル学習中にエラーが発生: {str(e)}")


if __name__ == "__main__":    
    try:
        session = create_session()

        # ローカルテスト実行の場合
        if len(sys.argv) > 1 and sys.argv[1] == "--local":
            result = training_sproc(session)
            print(f"Training results: {result}")
        # sprocの登録
        else:
            sproc_config = {
                "name": "TRAINING",
                "is_permanent": True,
                "stage_location": "@practice.ml.sproc",
                "packages": [
                    "snowflake-snowpark-python",
                    "scikit-learn",
                    "pandas",
                    "numpy"
                ],
                "imports": [
                    (os.path.join(IMPORTS_DIR, "utils/snowflake.py"), "src.utils.snowflake"),
                    (os.path.join(IMPORTS_DIR, "data/loader.py"), "src.data.loader"),
                    (os.path.join(IMPORTS_DIR, "data/preprocessing.py"), "src.data.preprocessing"),
                    (os.path.join(IMPORTS_DIR, "models/trainer.py"), "src.models.trainer"),
                    (os.path.join(IMPORTS_DIR, "evaluation/metrics.py"), "src.evaluation.metrics"),
                    (os.path.join(IMPORTS_DIR, "utils/config.py"), "src.utils.config"),

                    # ローカルリソースは sys._xoptions["snowflake_import_directory"] 以下にアップロードされる
                    os.path.join(IMPORTS_DIR, "config.yml"),
                    os.path.join(BASE_DIR, "connection_parameters.json")
                ],
                "replace": True,
                "execute_as": "caller"
            }

            session.sproc.register(
                func = training_sproc,
                **sproc_config
            )

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
