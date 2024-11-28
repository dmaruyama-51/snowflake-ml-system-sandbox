import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.models.trainer import train_model


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを生成"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        # 数値特徴量
        "ADMINISTRATIVE": np.random.randint(0, 10, n_samples),
        "ADMINISTRATIVE_DURATION": np.random.uniform(0, 100, n_samples),
        "INFORMATIONAL": np.random.randint(0, 10, n_samples),
        "INFORMATIONAL_DURATION": np.random.uniform(0, 100, n_samples),
        "PRODUCTRELATED": np.random.randint(0, 20, n_samples),
        "PRODUCTRELATED_DURATION": np.random.uniform(0, 200, n_samples),
        "BOUNCERATES": np.random.uniform(0, 1, n_samples),
        "EXITRATES": np.random.uniform(0, 1, n_samples),
        "PAGEVALUES": np.random.uniform(0, 100, n_samples),
        "SPECIALDAY": np.random.uniform(0, 1, n_samples),
        
        # カテゴリ特徴量
        "MONTH": np.random.choice(["Jan", "Feb", "Mar"], n_samples),
        "BROWSER": np.random.choice(["Chrome", "Firefox", "Safari"], n_samples),
        "REGION": np.random.choice(["A", "B", "C"], n_samples),
        "TRAFFICTYPE": np.random.randint(1, 4, n_samples),
        "VISITORTYPE": np.random.choice(["New", "Returning"], n_samples),
        "WEEKEND": np.random.choice([True, False], n_samples),
        
        # 目的変数
        "REVENUE": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)


def test_train_model_basic(sample_data, caplog):
    """基本的な学習プロセスのテスト"""
    # 少ない分割数で実行して時間を節約
    model, cv_scores = train_model(sample_data, n_splits=2, random_state=42)
    
    # 戻り値の型チェック
    assert isinstance(model, Pipeline)
    assert isinstance(cv_scores, list)
    assert len(cv_scores) == 2  # n_splits=2なので
    
    # メトリクスの確認
    expected_metrics = {"Accuracy", "Precision", "Recall", "ROC-AUC", "PR-AUC"}
    assert set(cv_scores[0].keys()) == expected_metrics


def test_train_model_predictions(sample_data):
    """学習したモデルの予測機能のテスト"""
    model, _ = train_model(sample_data, n_splits=2, random_state=42)
    
    # 予測の実行
    X = sample_data.drop("REVENUE", axis=1)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # 予測結果の検証
    assert len(predictions) == len(sample_data)
    assert all(pred in [0, 1] for pred in predictions)
    assert probabilities.shape == (len(sample_data), 2)
    assert all(0 <= prob <= 1 for prob in probabilities.flatten())


def test_train_model_reproducibility(sample_data):
    """再現性のテスト"""
    # 同じパラメータで2回実行
    model1, scores1 = train_model(sample_data, n_splits=2, random_state=42)
    model2, scores2 = train_model(sample_data, n_splits=2, random_state=42)
    
    # 予測結果の比較
    X = sample_data.drop("REVENUE", axis=1)
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    
    np.testing.assert_array_equal(pred1, pred2)
    assert scores1 == scores2


def test_train_model_invalid_data():
    """不正なデータでのテスト"""
    # 空のデータフレーム
    empty_df = pd.DataFrame()
    with pytest.raises(Exception):
        train_model(empty_df)
    
    # 必要な列が欠けているデータ
    invalid_df = pd.DataFrame({"ADMINISTRATIVE": [1, 2, 3]})
    with pytest.raises(Exception):
        train_model(invalid_df)
