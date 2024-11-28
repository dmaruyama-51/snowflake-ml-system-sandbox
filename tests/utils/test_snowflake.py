import json

import pytest
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSessionException, SnowparkSQLException
from snowflake.snowpark.dataframe import DataFrame as SnowparkDataFrame

from src.utils.snowflake import create_session, upload_dataframe_to_snowflake, fetch_dataframe_from_snowflake

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


@pytest.fixture
def mock_snowpark_df(mocker):
    """Snowpark DataFrameのモックを作成"""
    mock_df = mocker.MagicMock()
    mock_write = mocker.MagicMock()
    mock_mode = mocker.MagicMock()
    
    # モックチェーンの設定
    mock_df.write = mock_write
    mock_write.mode = mock_mode
    mock_mode.return_value = mock_mode
    mock_mode.save_as_table = mocker.MagicMock()
    
    return mock_df


@pytest.fixture
def mock_snowflake_session(mocker):
    """Snowflakeセッションのモックを作成"""
    mock_session = mocker.MagicMock()
    
    # 正常系のレスポンスを設定
    mock_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    mock_session.sql.return_value.to_pandas.return_value = mock_df
    
    return mock_session


def test_create_session_success(mock_json_file, mock_session_builder):
    """セッションが正常に作成される場合"""
    # 関数の実行
    result = create_session()

    # アサーション
    assert isinstance(result, Session)
    mock_session_builder.configs.assert_called_once_with(TEST_CONNECTION_PARAMS)


def test_create_session_connection_error(mock_json_file, mock_session_builder):
    """Snowflakeへの接続に失敗する場合"""
    # モックの設定
    mock_session_builder.configs.return_value.create.side_effect = (
        SnowparkSessionException("接続エラー")
    )

    # 例外が発生することを確認
    with pytest.raises(SnowparkSessionException) as exc_info:
        create_session()

    assert "接続エラー" in str(exc_info.value)


def test_upload_dataframe_to_snowflake_success(mock_snowflake_session, mock_snowpark_df):
    """データフレームが正常にアップロードされる場合"""
    # モックの設定
    mock_snowflake_session.create_dataframe.return_value = mock_snowpark_df
    
    # テストデータの準備
    test_df = pd.DataFrame({"col1": [1, 2, 3]})
    test_params = {
        "database_name": "test_db",
        "schema_name": "test_schema",
        "table_name": "test_table",
    }
    
    upload_dataframe_to_snowflake(
        session=mock_snowflake_session,
        df=test_df,
        **test_params
    )

    # アサーション
    expected_table_name = f"{test_params['database_name']}.{test_params['schema_name']}.{test_params['table_name']}"
    
    # データベースとスキーマの設定確認
    mock_snowflake_session.use_database.assert_called_once_with(test_params["database_name"])
    mock_snowflake_session.use_schema.assert_called_once_with(test_params["schema_name"])
    mock_snowflake_session.create_dataframe.assert_called_once_with(test_df)
    mock_snowpark_df.write.mode.return_value.save_as_table.assert_called_once_with(expected_table_name)


def test_upload_dataframe_to_snowflake_append_mode(mock_snowflake_session, mock_snowpark_df):
    """appendモードでデータフレームをアップロードする場合"""
    # モックの設定
    mock_snowflake_session.create_dataframe.return_value = mock_snowpark_df
    
    # テストデータの準備
    test_df = pd.DataFrame({"col1": [1, 2, 3]})
    test_params = {
        "database_name": "test_db",
        "schema_name": "test_schema",
        "table_name": "test_table",
    }

    upload_dataframe_to_snowflake(
        session=mock_snowflake_session,
        df=test_df,
        mode="append",
        **test_params
    )

    expected_table_name = f"{test_params['database_name']}.{test_params['schema_name']}.{test_params['table_name']}"
    
    # アサーション
    mock_snowflake_session.use_database.assert_called_once_with(test_params["database_name"])
    mock_snowflake_session.use_schema.assert_called_once_with(test_params["schema_name"])
    mock_snowflake_session.create_dataframe.assert_called_once_with(test_df)
    mock_snowpark_df.write.mode.assert_called_once_with("append")
    mock_snowpark_df.write.mode.return_value.save_as_table.assert_called_once_with(expected_table_name)


def test_upload_dataframe_to_snowflake_error(mock_snowflake_session):
    """Snowflakeへのアップロードに失敗する場合"""
    test_df = pd.DataFrame({"col1": [1, 2, 3]})
    error_message = "データベースエラー"
    mock_snowflake_session.use_database.side_effect = Exception(error_message)

    with pytest.raises(Exception) as exc_info:
        upload_dataframe_to_snowflake(
            session=mock_snowflake_session,
            df=test_df,
            database_name="test_db",
            schema_name="test_schema",
            table_name="test_table",
        )

    assert error_message in str(exc_info.value)


def test_fetch_dataframe_success(mock_snowflake_session):
    """正常系のテスト"""
    query = "SELECT * FROM test_table"
    df = fetch_dataframe_from_snowflake(mock_snowflake_session, query)
    
    # 戻り値の検証
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['col1', 'col2']
    
    # セッションの呼び出し確認
    mock_snowflake_session.sql.assert_called_once_with(query)

def test_fetch_dataframe_empty_query(mock_snowflake_session):
    """空クエリのテスト"""
    with pytest.raises(ValueError) as exc_info:
        fetch_dataframe_from_snowflake(mock_snowflake_session, "")
    
    assert "クエリが空です" in str(exc_info.value)


def test_fetch_dataframe_sql_error(mock_snowflake_session):
    """SQLエラーのテスト"""
    # SQLエラーを発生させる設定
    mock_snowflake_session.sql.side_effect = SnowparkSQLException("Invalid SQL")
    
    with pytest.raises(SnowparkSQLException) as exc_info:
        fetch_dataframe_from_snowflake(mock_snowflake_session, "SELECT * FROM invalid_table")
    
    assert "Invalid SQL" in str(exc_info.value)


def test_fetch_dataframe_general_error(mock_snowflake_session):
    """一般的なエラーのテスト"""
    # 一般的なエラーを発生させる設定
    mock_snowflake_session.sql.side_effect = Exception("Unexpected error")
    
    with pytest.raises(Exception) as exc_info:
        fetch_dataframe_from_snowflake(mock_snowflake_session, "SELECT * FROM test_table")
    
    assert "Unexpected error" in str(exc_info.value)
