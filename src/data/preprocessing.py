from typing import Tuple
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """データをテストデータとそれ以外とに分割"""
    test_size = 0.2
    random_state = 0

    logger.info(f"データの分割を開始 (test_size: {test_size})")
    logger.debug(f"入力データのサイズ: {df.shape}")

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"データ分割完了: 学習検証データ {train_val.shape}, テストデータ {test.shape}"
    )
    return train_val, test


def create_preprocessor() -> ColumnTransformer:
    """特徴量の前処理パイプラインを作成"""
    numeric_features = config["data"]["features"]["numeric"]
    categorical_features = config["data"]["features"]["categorical"]

    logger.info("前処理パイプラインの作成を開始")

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
        ]
    )

    logger.info("前処理パイプラインの作成完了")
    return preprocessor