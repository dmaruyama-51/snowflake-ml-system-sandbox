import pandas as pd
import numpy as np
from src.data.preprocessing import split_data, create_preprocessor
from sklearn.compose import ColumnTransformer


def test_split_data():
    # テスト用のダミーデータを作成
    dummy_data = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "target": np.random.randint(0, 2, 100),
        }
    )

    # 関数を実行
    train_val, test = split_data(dummy_data)

    # テストケース
    assert isinstance(
        train_val, pd.DataFrame
    ), "train_valはDataFrameである必要があります"
    assert isinstance(test, pd.DataFrame), "testはDataFrameである必要があります"

    # サイズの検証（test_size=0.2なので、8:2の分割になるはず）
    assert len(train_val) == 80, "train_valのサイズが期待値と異なります"
    assert len(test) == 20, "testのサイズが期待値と異なります"

    # 元のデータと分割後のデータの合計行数が同じことを確認
    assert len(dummy_data) == len(train_val) + len(
        test
    ), "分割後のデータ数が元のデータ数と一致しません"

def test_create_preprocessor():
    """create_preprocessor関数の基本的なテスト"""
    preprocessor = create_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer), "返り値がColumnTransformerではありません" 

def test_create_preprocessor_transformers():
    """前処理パイプラインの構成要素をテスト"""
    preprocessor = create_preprocessor()
    transformers = preprocessor.transformers

    # 数値特徴量とカテゴリ特徴量の変換器が存在することを確認
    transformer_names = [name for name, _, _ in transformers]
    assert "num" in transformer_names, "数値特徴量の変換器が見つかりません"
    assert "cat" in transformer_names, "カテゴリ特徴量の変換器が見つかりません"