import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import split_data, create_preprocessor
from src.utils.config import load_config

config = load_config()
NUMERIC_FEATURES = config["data"]["features"]["numeric"]
CATEGORICAL_FEATURES = config["data"]["features"]["categorical"]


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを生成"""
    np.random.seed(42)
    n_samples = 100

    # 数値特徴量の生成
    numeric_data = {
        feature: np.random.normal(0, 1, n_samples) for feature in NUMERIC_FEATURES
    }

    # カテゴリカル特徴量の生成
    categorical_data = {
        "MONTH": np.random.choice(["Jan", "Feb", "Mar"], n_samples),
        "BROWSER": np.random.choice(["Chrome", "Firefox"], n_samples),
        "REGION": np.random.choice(["A", "B", "C"], n_samples),
        "TRAFFICTYPE": np.random.choice([1, 2, 3], n_samples),
        "VISITORTYPE": np.random.choice(["New", "Returning"], n_samples),
        "WEEKEND": np.random.choice([True, False], n_samples),
    }

    # 目的変数の生成
    target = np.random.choice([0, 1], n_samples)

    # DataFrameの作成
    df = pd.DataFrame({**numeric_data, **categorical_data})
    df["REVENUE"] = target

    return df


def test_split_data(sample_data):
    """データ分割のテスト"""
    # 基本的な分割のテスト
    train_val, test = split_data(sample_data)

    # サイズの確認
    assert len(train_val) + len(test) == len(sample_data)
    assert len(test) == int(len(sample_data) * 0.2)

    # カラムの一致確認
    assert all(col in train_val.columns for col in sample_data.columns)
    assert all(col in test.columns for col in sample_data.columns)

    # データの重複がないことを確認
    assert len(pd.concat([train_val, test]).drop_duplicates()) == len(sample_data)


def test_create_preprocessor(sample_data):
    """前処理パイプラインのテスト"""
    preprocessor = create_preprocessor()

    # 特徴量の分割
    X = sample_data.drop("REVENUE", axis=1)

    # フィットと変換
    X_transformed = preprocessor.fit_transform(X)

    # 変換後のデータ形状の確認
    expected_features = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert X_transformed.shape[1] == expected_features

    # 数値特徴量のスケーリング確認
    numeric_means = np.mean(X_transformed[:, : len(NUMERIC_FEATURES)], axis=0)
    numeric_stds = np.std(X_transformed[:, : len(NUMERIC_FEATURES)], axis=0)

    # StandardScalerによる変換後は平均0、標準偏差1に近い値になるはず
    assert np.allclose(numeric_means, 0, atol=1e-10)
    assert np.allclose(numeric_stds, 1, atol=1e-10)


def test_preprocessor_with_unknown_categories(sample_data):
    """未知のカテゴリ値に対する処理のテスト"""
    preprocessor = create_preprocessor()

    # 学習データの準備
    X_train = sample_data.iloc[:80].drop("REVENUE", axis=1)
    X_test = sample_data.iloc[80:].drop("REVENUE", axis=1)

    # 未知のカテゴリ値を追加
    unknown_idx = X_test.index[0]
    X_test.loc[unknown_idx, "BROWSER"] = "Unknown_Browser"

    # フィットと変換
    preprocessor.fit(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 変換後のデータ形状の確認
    expected_features = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert X_test_transformed.shape[1] == expected_features

    # カテゴリカル特徴量のインデックスを取得
    cat_feature_indices = {
        feature: idx + len(NUMERIC_FEATURES)
        for idx, feature in enumerate(CATEGORICAL_FEATURES)
    }

    # 未知のカテゴリ値が-1にエンコードされていることを確認
    browser_idx = cat_feature_indices["BROWSER"]
    unknown_category_value = X_test_transformed[0, browser_idx]
    assert (
        unknown_category_value == -1
    ), f"未知のカテゴリ値のエンコーディング結果が期待値(-1)と異なります: {unknown_category_value}"

    # 既知のカテゴリ値が-1以外にエンコードされていることを確認
    known_category_values = X_test_transformed[1:, browser_idx]
    assert all(
        val != -1 for val in known_category_values
    ), "既知のカテゴリ値が-1にエンコードされています"

    # 他のカテゴリカル特徴量が正しくエンコードされていることを確認
    for feature in CATEGORICAL_FEATURES:
        if feature != "BROWSER":
            feature_idx = cat_feature_indices[feature]
            assert all(
                val != -1 for val in X_test_transformed[:, feature_idx]
            ), f"{feature}の値が不正にエンコードされています"
