from typing import List, Final, Tuple
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: Final[List[str]] = [
    "ADMINISTRATIVE",
    "ADMINISTRATIVE_DURATION",
    "INFORMATIONAL",
    "INFORMATIONAL_DURATION",
    "PRODUCTRELATED",
    "PRODUCTRELATED_DURATION",
    "BOUNCERATES",
    "EXITRATES",
    "PAGEVALUES",
    "SPECIALDAY",
]

CATEGORICAL_FEATURES: Final[List[str]] = [
    "MONTH",
    "BROWSER",
    "REGION",
    "TRAFFICTYPE",
    "VISITORTYPE",
    "WEEKEND",
]


def split_data(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """データをテストデータとそれ以外とに分割"""
    logger.info(f"データの分割を開始 (test_size: {test_size})")
    logger.debug(f"入力データのサイズ: {df.shape}")
    
    train_val, test = train_test_split(df, test_size=test_size, random_state=0)
    
    logger.info(f"データ分割完了: 学習検証データ {train_val.shape}, テストデータ {test.shape}")
    logger.debug(f"学習検証データのカラム: {', '.join(train_val.columns)}")
    
    return train_val, test


def create_preprocessor() -> ColumnTransformer:
    """特徴量の前処理パイプラインを作成"""
    logger.info("前処理パイプラインの作成を開始")
    logger.debug(f"数値特徴量: {', '.join(NUMERIC_FEATURES)}")
    logger.debug(f"カテゴリ特徴量: {', '.join(CATEGORICAL_FEATURES)}")
    
    preprocessor = ColumnTransformer(
        [
            # randomforestではscalingの必要はないが、別アルゴリズムへの拡張性のため入れておく
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    
    logger.info("前処理パイプラインの作成完了")
    
    return preprocessor
