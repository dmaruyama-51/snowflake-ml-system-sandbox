import pandas as pd
import pytest

from src.data.loader import fetch_dataset
from src.utils.config import load_config

config = load_config()


@pytest.fixture
def mock_snowflake_session(mocker):
    """Snowflakeセッションのモック"""
    session = mocker.Mock()

    # モックデータの作成
    mock_data = pd.DataFrame(
        {
            # カテゴリカル特徴量
            **{
                str(col): ["A"] * 5 for col in config["data"]["features"]["categorical"]
            },
            # 数値特徴量
            **{str(col): [1.0] * 5 for col in config["data"]["features"]["numeric"]},
            # ターゲット
            str(config["data"]["target"]): [0, 1, 0, 1, 0],
        }
    )

    # to_pandas()を返すSQLクエリの結果をモック
    query_result = mocker.Mock()
    query_result.to_pandas.return_value = mock_data
    session.sql.return_value = query_result

    return session


def test_fetch_dataset_training_mode(mock_snowflake_session):
    """学習モードでのデータ取得テスト"""
    # 実行
    df = fetch_dataset(mock_snowflake_session, is_training=True)

    # アサーション
    assert df is not None
    assert isinstance(df, pd.DataFrame)

    # SQLクエリに学習用の日付条件が含まれていることを確認
    sql_query = mock_snowflake_session.sql.call_args[0][0]
    start_date = config["data"]["period"]["start_date"]
    end_date = config["data"]["period"]["end_date"]
    assert f"BETWEEN '{start_date}' AND '{end_date}'" in sql_query


def test_fetch_dataset_inference_mode(mock_snowflake_session):
    """推論モードでのデータ取得テスト"""
    # 実行
    df = fetch_dataset(mock_snowflake_session, is_training=False)

    # アサーション
    assert df is not None
    assert isinstance(df, pd.DataFrame)

    # SQLクエリに推論用の固定日付が含まれていることを確認
    sql_query = mock_snowflake_session.sql.call_args[0][0]
    assert "SESSION_DATE = '2024-12-01'" in sql_query


def test_fetch_dataset_columns(mock_snowflake_session):
    """取得するカラムの確認テスト"""
    # 実行
    df = fetch_dataset(mock_snowflake_session, is_training=True)

    # 期待されるカラム（リストを展開して結合）
    expected_columns = (
        [str(col) for col in config["data"]["features"]["categorical"]]
        + [str(col) for col in config["data"]["features"]["numeric"]]
        + [str(config["data"]["target"])]
    )

    # アサーション
    assert set(df.columns) == set(expected_columns)


def test_fetch_dataset_error_handling(mocker):
    """エラーハンドリングのテスト"""
    # エラーを発生させるモックセッション
    error_session = mocker.Mock()
    error_session.sql.side_effect = Exception("Database connection failed")

    # エラーが発生することを確認
    with pytest.raises(RuntimeError) as exc_info:
        fetch_dataset(error_session, is_training=True)

    assert "データセット取得中にエラーが発生" in str(exc_info.value)
