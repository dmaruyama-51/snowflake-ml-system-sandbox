import json

import pytest
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSessionException

from src.dwh.snowflake import create_session

TEST_CONNECTION_PARAMS = {
    "account": "test_account",
    "user": "test_user",
    "password": "test_password",
    "warehouse": "test_warehouse",
    "database": "test_database",
    "schema": "test_schema",
}


@pytest.fixture
def mock_json_file(mocker):
    """設定ファイルのモックを作成するフィクスチャ"""
    mock_data = json.dumps(TEST_CONNECTION_PARAMS)
    return mocker.patch("builtins.open", mocker.mock_open(read_data=mock_data))


@pytest.fixture
def mock_session_builder(mocker):
    """Session.builderのモックを作成するフィクスチャ"""
    mock_session = mocker.MagicMock(spec=Session)
    mock_builder = mocker.MagicMock()
    mock_builder.configs.return_value.create.return_value = mock_session
    mocker.patch.object(Session, "builder", mock_builder)
    return mock_builder


def test_create_session_success(mock_json_file, mock_session_builder):
    """正常系のテスト：セッションが正常に作成される場合"""
    # 関数の実行
    result = create_session()

    # アサーション
    assert isinstance(result, Session)
    mock_session_builder.configs.assert_called_once_with(TEST_CONNECTION_PARAMS)


def test_create_session_connection_error(mock_json_file, mock_session_builder):
    """異常系のテスト：Snowflakeへの接続に失敗する場合"""
    # モックの設定
    mock_session_builder.configs.return_value.create.side_effect = (
        SnowparkSessionException("接続エラー")
    )

    # 例外が発生することを確認
    with pytest.raises(SnowparkSessionException) as exc_info:
        create_session()

    assert "接続エラー" in str(exc_info.value)
