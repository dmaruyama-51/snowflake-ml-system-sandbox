import pytest
import numpy as np
from src.evaluation.metrics import calc_evaluation_metrics


def test_calc_evaluation_metrics_perfect():
    """完璧な予測の場合のテスト"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.1, 0.9])

    metrics = calc_evaluation_metrics(y_true, y_pred, y_pred_proba)

    assert metrics["Accuracy"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["ROC-AUC"] == 1.0
    assert metrics["PR-AUC"] == 1.0


def test_calc_evaluation_metrics_empty():
    """空の入力データの場合のテスト"""
    with pytest.raises(ValueError):
        calc_evaluation_metrics(np.array([]), np.array([]), np.array([]))


def test_calc_evaluation_metrics_mismatched_shapes():
    """入力配列のサイズが一致しない場合のテスト"""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])  # サイズが異なる
    y_pred_proba = np.array([0.1, 0.9, 0.1])

    with pytest.raises(ValueError):
        calc_evaluation_metrics(y_true, y_pred, y_pred_proba)
