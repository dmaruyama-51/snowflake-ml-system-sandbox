import numpy as np
import pytest
from src.evaluation.metrics import calc_evaluation_metrics


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを生成"""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4])
    return y_true, y_pred, y_pred_proba


def test_calc_evaluation_metrics_success(sample_data):
    """正常系のテスト"""
    y_true, y_pred, y_pred_proba = sample_data
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
