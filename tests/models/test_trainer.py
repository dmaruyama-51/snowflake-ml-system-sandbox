import pytest
import pandas as pd
import numpy as np
from src.models.trainer import train_model
from src.utils.config import load_config
from sklearn.pipeline import Pipeline

config = load_config()

@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを作成"""
    np.random.seed(42)
    n_samples = 100

    numeric_features = config["data"]["features"]["numeric"]
    categorical_features = config["data"]["features"]["categorical"]
    
    # 実際のデータ構造に合わせたカラムを追加
    X = pd.DataFrame({
        # numeric features
        **{col: np.random.uniform(0, 100, n_samples) for col in numeric_features},
        # categorical features
        **{col: np.random.choice(['A', 'B', 'C'], n_samples) for col in categorical_features}
    })
    y = pd.Series(np.random.randint(0, 2, n_samples), name='REVENUE')
    
    return pd.concat([X, y], axis=1)

def test_train_model_basic(sample_data):
    """train_modelの基本的な機能をテスト"""
    # 実行
    model, cv_scores = train_model(sample_data, n_splits=3, random_state=42)
    
    # 基本的なアサーション
    assert isinstance(model, Pipeline)
    assert len(cv_scores) == 3  # n_splits=3なので
    assert isinstance(cv_scores[0], dict)
    
    # メトリクス名を小文字に統一して確認
    expected_metrics = {'accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc'}
    actual_metrics = {k.lower().replace('-', '_') for k in cv_scores[0].keys()}
    assert expected_metrics.issubset(actual_metrics)

def test_train_model_predictions(sample_data):
    """学習したモデルの予測機能をテスト"""
    # モデルの学習
    model, _ = train_model(sample_data, n_splits=3, random_state=42)
    
    # テストデータで予測
    X_test = sample_data.drop('REVENUE', axis=1)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # 予測結果の検証
    assert len(predictions) == len(sample_data)
    assert all(pred in [0, 1] for pred in predictions)
    assert probabilities.shape == (len(sample_data), 2)
    assert all(0 <= prob <= 1 for prob in probabilities.flatten())