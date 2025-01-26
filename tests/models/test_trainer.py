import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.trainer import (
    calc_evaluation_metrics,
    create_model_pipeline,
    train_model,
)
from src.utils.config import load_config

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


def test_train_model_with_hyperparameter_optimization(sample_data):
    """ハイパーパラメータ最適化を含むモデル学習のテスト"""
    # 実行
    model, metrics = train_model(
        sample_data,
        n_splits=2,  # テスト用に少ない分割数
        random_state=42,
        optimize_hyperparams=True,
        n_trials=3,  # テスト用に少ない試行回数
    )

    # モデルの検証
    assert isinstance(model, Pipeline)

    # RandomForestClassifierのパラメータが最適化されているか確認
    rf_params = model.named_steps["classifier"].get_params()
    optimizable_params = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "criterion",
    }
    # 少なくとも1つのパラメータが最適化されていることを確認
    assert any(param in rf_params for param in optimizable_params)

    # メトリクスの検証
    assert isinstance(metrics, dict)
    expected_metrics = {"Accuracy", "Precision", "Recall", "ROC-AUC", "PR-AUC"}
    assert set(metrics.keys()) == expected_metrics


def test_train_model_without_hyperparameter_optimization(sample_data):
    """ハイパーパラメータ最適化なしのモデル学習のテスト"""
    # 実行
    model, metrics = train_model(
        sample_data, n_splits=2, random_state=42, optimize_hyperparams=False
    )

    # モデルの検証
    assert isinstance(model, Pipeline)

    # デフォルトパラメータが使用されているか確認
    rf_params = model.named_steps["classifier"].get_params()
    assert rf_params["random_state"] == 42
    # デフォルトパラメータ以外が変更されていないことを確認
    default_rf = RandomForestClassifier(random_state=42)
    for param, value in rf_params.items():
        if param != "random_state":
            assert value == default_rf.get_params()[param]


def test_objective_function(sample_data):
    """Optuna目的関数のテスト"""
    import optuna

    from src.models.trainer import objective

    # テストデータの準備
    X = sample_data.drop("REVENUE", axis=1)
    y = sample_data["REVENUE"]

    # Studyオブジェクトの作成
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    # 目的関数の実行
    score = objective(trial, X, y, n_splits=2, random_state=42)

    # スコアの検証
    assert isinstance(score, float)
    assert 0 <= score <= 1
