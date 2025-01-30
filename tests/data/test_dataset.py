import pandas as pd
import pytest

from src.data.dataset import create_ml_dataset, update_ml_dataset


@pytest.fixture
def mock_snowflake_session(mocker):
    mock_session = mocker.Mock()
    mock_session.sql.return_value.collect.return_value = None
    return mock_session


@pytest.fixture
def mock_snowflake_data():
    return pd.DataFrame(
        {
            "UID": [1, 2],
            "SESSION_DATE": ["2024-03-20", "2024-03-20"],
            "REVENUE": [1, 0],
            "ADMINISTRATIVE": [0, 1],
            "ADMINISTRATIVE_DURATION": [0.0, 10.0],
            "INFORMATIONAL": [0, 1],
            "INFORMATIONAL_DURATION": [0.0, 5.0],
            "PRODUCTRELATED": [2, 3],
            "PRODUCTRELATED_DURATION": [20.0, 30.0],
            "BOUNCERATES": [0.2, 0.1],
            "EXITRATES": [0.1, 0.2],
            "PAGEVALUES": [0.0, 10.0],
            "SPECIALDAY": [0, 0],
            "OPERATINGSYSTEMS": [1, 2],
            "BROWSER": [1, 2],
            "REGION": [1, 2],
            "TRAFFICTYPE": [1, 2],
            "VISITORTYPE": ["New_Visitor", "Returning_Visitor"],
            "WEEKEND": [0, 1],
        }
    )


def test_create_ml_dataset_success(mock_snowflake_session):
    """create_ml_datasetの正常系テスト"""
    create_ml_dataset(
        session=mock_snowflake_session,
        target_date="2024-03-20",
        database_name="TEST_DB",
        schema_name="TEST_SCHEMA",
        table_name="dataset",
        source_table_name="online_shoppers_intention",
    )

    # SQLが実行されたことを確認
    mock_snowflake_session.sql.assert_called_once()
    # 実行されたSQLにテーブル名とターゲット日付が含まれていることを確認
    sql_call = mock_snowflake_session.sql.call_args[0][0]
    assert "TEST_DB.TEST_SCHEMA.dataset" in sql_call
    assert "session_date <= '2024-03-20'" in sql_call


def test_update_ml_dataset_with_data(
    mock_snowflake_session, mock_snowflake_data, mocker
):
    """update_ml_datasetの正常系テスト（データあり）"""
    # to_pandasの戻り値をモック
    mock_snowflake_session.sql.return_value.to_pandas.return_value = mock_snowflake_data

    # upload_dataframe_to_snowflakeをモック
    mock_upload = mocker.patch("src.data.dataset.upload_dataframe_to_snowflake")

    update_ml_dataset(
        session=mock_snowflake_session,
        target_date="2024-03-20",
        database_name="TEST_DB",
        schema_name="TEST_SCHEMA",
        table_name="dataset",
        source_table_name="online_shoppers_intention",
    )

    # SQLが実行されたことを確認
    mock_snowflake_session.sql.assert_called_once()
    # アップロード関数が呼ばれたことを確認
    mock_upload.assert_called_once()


def test_update_ml_dataset_no_data(mock_snowflake_session, mocker):
    """update_ml_datasetの正常系テスト（データなし）"""
    # 空のDataFrameを返すようにモック
    mock_snowflake_session.sql.return_value.to_pandas.return_value = pd.DataFrame()

    # upload_dataframe_to_snowflakeをモック
    mock_upload = mocker.patch("src.data.dataset.upload_dataframe_to_snowflake")

    update_ml_dataset(
        session=mock_snowflake_session,
        target_date="2024-03-20",
        database_name="TEST_DB",
        schema_name="TEST_SCHEMA",
    )

    # SQLは実行されるが、データがないためアップロードは実行されないことを確認
    mock_snowflake_session.sql.assert_called_once()
    mock_upload.assert_not_called()


def test_create_ml_dataset_error(mock_snowflake_session):
    """create_ml_datasetのエラー系テスト"""
    mock_snowflake_session.sql.side_effect = Exception("テストエラー")

    with pytest.raises(Exception) as exc_info:
        create_ml_dataset(
            session=mock_snowflake_session,
            target_date="2024-03-20",
            database_name="TEST_DB",
            schema_name="TEST_SCHEMA",
        )

    assert str(exc_info.value) == "テストエラー"
