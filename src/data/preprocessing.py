from typing import Tuple
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


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
