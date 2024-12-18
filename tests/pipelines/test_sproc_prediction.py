import pandas as pd
import pytest
from snowflake.snowpark import Session

from src.pipelines.sproc_prediction import sproc_prediction


def test_sproc_prediction_success(mocker):
    # モックセッションの作成
    mock_session = mocker.Mock(spec=Session)
    
    # 各依存関数のモック化
    mock_fetch = mocker.patch('src.pipelines.sproc_prediction.fetch_dataset')
    mock_fetch.return_value = pd.DataFrame({
        'UID': [1, 2],
        'FEATURE1': [0.1, 0.2]
    })
    
    mock_load_model = mocker.patch('src.pipelines.sproc_prediction.load_latest_model')
    mock_model_version = mocker.Mock()
    mock_model_version._model_name = "test_model"
    mock_model_version._version_name = "v1"
    mock_load_model.return_value = (mock_model_version, mocker.Mock())
    
    mock_predict = mocker.patch('src.pipelines.sproc_prediction.predict')
    mock_predict.return_value = [0.8, 0.9]
    
    mock_upload = mocker.patch('src.pipelines.sproc_prediction.upload_dataframe_to_snowflake')
    
    # テスト実行
    result = sproc_prediction(mock_session, "2024-03-20")
    
    # アサーション
    assert result == 1
    mock_fetch.assert_called_once()
    mock_load_model.assert_called_once()
    mock_predict.assert_called_once()
    mock_upload.assert_called_once()
    

def test_sproc_prediction_fetch_dataset_returns_none(mocker):
    mock_session = mocker.Mock(spec=Session)
    mock_fetch = mocker.patch('src.pipelines.sproc_prediction.fetch_dataset')
    mock_fetch.return_value = None
    
    with pytest.raises(SystemExit) as exc_info:
        sproc_prediction(mock_session, "2024-03-20")
    
    assert exc_info.value.code == 1


