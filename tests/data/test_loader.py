import pytest
import pandas as pd
from snowflake.snowpark import Session
from src.data.loader import fetch_dataset

@pytest.fixture
def mock_session(mocker):
    """Snowflakeセッションのモック"""
    mock = mocker.Mock(spec=Session)
    return mock

def test_fetch_dataset_success(mock_session):
    """正常系: データセット取得が成功するケース"""
    # モックデータの準備
    expected_df = pd.DataFrame({
        'categorical_col': ['A', 'B'],
        'numerical_col': [1, 2],
        'target': [0, 1]
    })
    
    # セッションのSQL実行結果をモック
    mock_session.sql.return_value.to_pandas.return_value = expected_df
    
    # 関数実行
    result = fetch_dataset(mock_session)
    
    # アサーション
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    mock_session.sql.assert_called_once()

def test_fetch_dataset_error(mock_session):
    """異常系: データセット取得でエラーが発生するケース"""
    # セッションのSQL実行でエラーを発生させる
    mock_session.sql.side_effect = Exception("データベース接続エラー")
    
    # エラーが発生することを確認
    with pytest.raises(RuntimeError) as exc_info:
        fetch_dataset(mock_session)
    
    assert "データセット取得中にエラーが発生" in str(exc_info.value)
    mock_session.sql.assert_called_once()

def test_fetch_dataset_empty(mock_session):
    """境界値: 空のデータフレームが返されるケース"""
    # 空のデータフレームを返すようにモック
    mock_session.sql.return_value.to_pandas.return_value = pd.DataFrame()
    
    # 関数実行
    result = fetch_dataset(mock_session)
    
    # アサーション
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    mock_session.sql.assert_called_once()