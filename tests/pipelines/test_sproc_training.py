import numpy as np
import pandas as pd
import pytest
from snowflake.snowpark import Session

from src.pipelines.sproc_training import sproc_training


def test_sproc_training_success(mocker):
    # モックセッションの作成
    mock_session = mocker.Mock(spec=Session)

    # 各依存関数のモック化
    mock_fetch = mocker.patch("src.pipelines.sproc_training.fetch_training_dataset")
    mock_fetch.return_value = pd.DataFrame(
        {
            "UID": ["user1", "user2"],
            "FEATURE1": [0.1, 0.2],
            "REVENUE": [0, 1],
        }
    )

    mock_split = mocker.patch("src.pipelines.sproc_training.split_data")
    mock_split.return_value = (
        pd.DataFrame({"FEATURE1": [0.1, 0.2], "REVENUE": [0, 1]}),
        pd.DataFrame({"FEATURE1": [0.3, 0.4], "REVENUE": [0, 1]}),
    )

    mock_train = mocker.patch("src.pipelines.sproc_training.train_model")
    mock_pipeline = mocker.Mock()
    mock_pipeline.predict.return_value = np.array([0, 1])  # 両方のクラスを予測
    mock_pipeline.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7]]
    )  # 各クラスの確率を設定
    mock_train.return_value = (mock_pipeline, ([{"accuracy": 0.85}], None))

    mock_registry = mocker.Mock()
    mocker.patch("src.pipelines.sproc_training.Registry", return_value=mock_registry)

    # テスト実行
    result = sproc_training(mock_session)

    # アサーション
    assert result == 1
    mock_fetch.assert_called_once()
    mock_split.assert_called_once()
    mock_train.assert_called_once()
    mock_registry.log_model.assert_called_once()


def test_sproc_training_fetch_dataset_returns_none(mocker):
    mock_session = mocker.Mock(spec=Session)
    mock_fetch = mocker.patch("src.pipelines.sproc_training.fetch_training_dataset")
    mock_fetch.return_value = None

    with pytest.raises(ValueError, match="Failed to fetch dataset"):
        sproc_training(mock_session)

    # fetch_datasetが呼び出されたことを確認
    mock_fetch.assert_called_once()
