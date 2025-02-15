import numpy as np
import pandas as pd
import pytest
from snowflake.ml.model import ModelVersion
from snowflake.ml.registry import Registry

from src.models.predictor import load_latest_model_version, load_default_model_version, predict_proba, predict_label


@pytest.fixture
def mock_registry(mocker):
    """Registry モックを作成するフィクスチャ"""
    mock_registry = mocker.Mock(spec=Registry)

    # get_model の戻り値を設定
    mock_model_ref = mocker.Mock()
    mock_model_version = mocker.Mock(spec=ModelVersion)

    mock_model_ref.last.return_value = mock_model_version
    mock_model_ref.default = mock_model_version
    mock_registry.get_model.return_value = mock_model_ref

    return mock_registry


def test_load_latest_model_version(mocker, mock_registry):
    """load_latest_model_version のテスト"""
    # Registry クラスのモックを設定
    mocker.patch("src.models.predictor.Registry", return_value=mock_registry)

    # テスト実行
    session = mocker.Mock()
    model_version = load_latest_model_version(session)

    # アサーション
    assert isinstance(model_version, mocker.Mock)
    mock_registry.get_model.assert_called_once_with("random_forest")
    
    # last メソッドが呼ばれたことを確認
    model_ref = mock_registry.get_model.return_value
    model_ref.last.assert_called_once()


def test_load_default_model_version(mocker, mock_registry):
    """load_default_model_version のテスト"""
    # Registry クラスのモックを設定
    mocker.patch("src.models.predictor.Registry", return_value=mock_registry)

    # テスト実行
    session = mocker.Mock()
    model_version = load_default_model_version(session)

    # アサーション
    assert isinstance(model_version, mocker.Mock)
    mock_registry.get_model.assert_called_once_with("random_forest")
    
    # default プロパティにアクセスしたことを確認
    model_ref = mock_registry.get_model.return_value
    assert model_ref.default == model_version


def test_predict_proba(mocker):
    """predict_proba のテスト"""
    # テストデータ作成
    features = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})

    # ModelVersion モックの作成
    mock_model_version = mocker.Mock(spec=ModelVersion)
    mock_predictions = pd.DataFrame(
        {"output_feature_0": [0.2, 0.3, 0.4], "output_feature_1": [0.8, 0.7, 0.6]}
    )
    mock_model_version.run.return_value = mock_predictions

    # テスト実行
    predictions = predict_proba(features, mock_model_version)

    # アサーション
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3
    np.testing.assert_array_almost_equal(predictions, np.array([0.8, 0.7, 0.6]))
    mock_model_version.run.assert_called_once_with(features, function_name="predict_proba")


def test_predict_label(mocker):
    """predict_label のテスト"""
    # テストデータ作成
    features = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})

    # ModelVersion モックの作成
    mock_model_version = mocker.Mock(spec=ModelVersion)
    mock_predictions = pd.DataFrame(
        {"output_feature_0": [1, 0, 1]}
    )
    mock_model_version.run.return_value = mock_predictions

    # テスト実行
    predictions = predict_label(features, mock_model_version)

    # アサーション
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3
    np.testing.assert_array_almost_equal(predictions, np.array([1, 0, 1]))
    mock_model_version.run.assert_called_once_with(features, function_name="predict")
