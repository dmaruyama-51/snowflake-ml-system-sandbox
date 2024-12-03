import pytest
import pandas as pd
import numpy as np
from src.models.trainer import (
    train_model,
    create_model_pipeline,
    calc_evaluation_metrics,
)
from src.utils.config import load_config
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

config = load_config()


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを作成"""
    np.random.seed(42)
    n_samples = 100

    numeric_features = config["data"]["features"]["numeric"]
    categorical_features = config["data"]["features"]["categorical"]

    # 実際のデータ構造に合わせたカラムを追加
    X = pd.DataFrame(
        {
            # numeric features
            **{col: np.random.uniform(0, 100, n_samples) for col in numeric_features},
            # categorical features
            **{
                col: np.random.choice(["A", "B", "C"], n_samples)
                for col in categorical_features
            },
        }
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), name="REVENUE")

    return pd.concat([X, y], axis=1)


def test_create_model_pipeline():
    """モデルパイプライン作成のテスト"""
    # パイプラインの作成
    pipeline = create_model_pipeline(random_state=42)

    # 戻り値の型チェック
    assert isinstance(pipeline, Pipeline), "戻り値がPipelineインスタンスではありません"

    # パイプラインのステップ数チェック
    assert len(pipeline.steps) == 2, "パイプラインのステップ数が想定と異なります"

    # 各ステップの名前と型のチェック
    preprocessor_step = pipeline.steps[0]
    classifier_step = pipeline.steps[1]

    # 前処理ステップのチェック
    assert preprocessor_step[0] == "preprocessor", "前処理ステップの名前が不正です"
    assert isinstance(
        preprocessor_step[1], ColumnTransformer
    ), "前処理ステップの型が不正です"

    # 分類器ステップのチェック
    assert classifier_step[0] == "classifier", "分類器ステップの名前が不正です"
    assert isinstance(
        classifier_step[1], RandomForestClassifier
    ), "分類器ステップの型が不正です"


def test_train_model_basic(sample_data):
    """train_modelの基本的な機能をテスト"""
    # 実行
    model, cv_scores = train_model(sample_data, n_splits=3, random_state=42)

    # 基本的なアサーション
    assert isinstance(model, Pipeline)
    assert len(cv_scores) == 3  # n_splits=3なので
    assert isinstance(cv_scores[0], dict)

    # メトリクス名を小文字に統一して確認
    expected_metrics = {"accuracy", "precision", "recall", "roc_auc", "pr_auc"}
    actual_metrics = {k.lower().replace("-", "_") for k in cv_scores[0].keys()}
    assert expected_metrics.issubset(actual_metrics)


def test_train_model_predictions(sample_data):
    """学習したモデルの予測機能をテスト"""
    # モデルの学習
    model, _ = train_model(sample_data, n_splits=3, random_state=42)

    # テストデータで予測
    X_test = sample_data.drop("REVENUE", axis=1)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # 予測結果の検証
    assert len(predictions) == len(sample_data)
    assert all(pred in [0, 1] for pred in predictions)
    assert probabilities.shape == (len(sample_data), 2)
    assert all(0 <= prob <= 1 for prob in probabilities.flatten())


@pytest.fixture
def sample_metrics_data():
    """テスト用のサンプルデータを生成"""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4])
    return y_true, y_pred, y_pred_proba


def test_calc_evaluation_metrics_success(sample_metrics_data):
    """正常系のテスト"""
    y_true, y_pred, y_pred_proba = sample_metrics_data
    metrics = calc_evaluation_metrics(y_true, y_pred, y_pred_proba)

    # 必要なメトリクスが全て含まれているか確認
    expected_metrics = {"Accuracy", "Precision", "Recall", "ROC-AUC", "PR-AUC"}
    assert set(metrics.keys()) == expected_metrics

    # 各メトリクスが適切な範囲（0-1）に収まっているか確認
    for value in metrics.values():
        assert 0 <= value <= 1


def test_calc_evaluation_metrics_invalid_input():
    """異常系: 不正な入力データでのテスト"""
    y_true = np.array([0, 1])
    y_pred = np.array([0])  # サイズが異なる
    y_pred_proba = np.array([0.5])

    with pytest.raises(ValueError):
        calc_evaluation_metrics(y_true, y_pred, y_pred_proba)
