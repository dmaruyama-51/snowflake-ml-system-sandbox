import pandas as pd
import pytest

from src.data.loader import fetch_training_dataset, fetch_prediction_dataset, fetch_test_dataset
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


def test_fetch_training_dataset(mock_snowflake_session, mocker):
    """学習用データセット取得のテスト"""
    # 現在時刻をモック
    mock_now = pd.Timestamp("2024-03-15")
    mocker.patch("pandas.Timestamp.now", return_value=mock_now)

    # 実行
    df = fetch_training_dataset(mock_snowflake_session)

    # アサーション
    assert df is not None
    assert isinstance(df, pd.DataFrame)

    # SQLクエリに学習用の日付条件が含まれていることを確認
    sql_query = mock_snowflake_session.sql.call_args[0][0]
    period_months = config["data"]["period"]["months"]
    expected_end_date = mock_now.strftime("%Y-%m-%d")
    expected_start_date = (mock_now - pd.DateOffset(months=period_months)).strftime("%Y-%m-%d")
    assert f"BETWEEN '{expected_start_date}' AND '{expected_end_date}'" in sql_query


def test_fetch_prediction_dataset(mock_snowflake_session):
    """推論用データセット取得のテスト"""
    # 実行
    prediction_date = "2024-12-01"
    df = fetch_prediction_dataset(mock_snowflake_session, prediction_date=prediction_date)

    # アサーション
    assert df is not None
    assert isinstance(df, pd.DataFrame)

    # SQLクエリに推論用の日付が含まれていることを確認
    sql_query = mock_snowflake_session.sql.call_args[0][0]
    assert f"SESSION_DATE = '{prediction_date}'" in sql_query


def test_fetch_dataset_columns(mock_snowflake_session, mocker):
    """取得するカラムの確認テスト"""
    # モックデータにUIDカラムを追加
    mock_data = pd.DataFrame(
        {
            "UID": ["1"] * 5,  # UIDカラムを追加
            **{
                str(col): ["A"] * 5 for col in config["data"]["features"]["categorical"]
            },
            **{str(col): [1.0] * 5 for col in config["data"]["features"]["numeric"]},
            str(config["data"]["target"]): [0, 1, 0, 1, 0],
        }
    )

    # モックの戻り値を更新
    query_result = mocker.Mock()
    query_result.to_pandas.return_value = mock_data
    mock_snowflake_session.sql.return_value = query_result

    # 実行（学習用データセットで確認）
    df = fetch_training_dataset(mock_snowflake_session)

    # 期待されるカラム
    expected_columns = (
        ["UID"]
        + [str(col) for col in config["data"]["features"]["categorical"]]
        + [str(col) for col in config["data"]["features"]["numeric"]]
        + [str(config["data"]["target"])]
    )

    # アサーション
    assert set(df.columns) == set(expected_columns)


def test_fetch_training_dataset_error(mocker):
    """学習用データセット取得時のエラーハンドリングテスト"""
    # エラーを発生させるモックセッション
    error_session = mocker.Mock()
    error_session.sql.side_effect = Exception("Database connection failed")

    # エラーが発生することを確認
    with pytest.raises(RuntimeError) as exc_info:
        fetch_training_dataset(error_session)

    assert "Error occurred during training dataset retrieval" in str(exc_info.value)


def test_fetch_prediction_dataset_error(mocker):
    """推論用データセット取得時のエラーハンドリングテスト"""
    # エラーを発生させるモックセッション
    error_session = mocker.Mock()
    error_session.sql.side_effect = Exception("Database connection failed")

    # エラーが発生することを確認
    with pytest.raises(RuntimeError) as exc_info:
        fetch_prediction_dataset(error_session, prediction_date="2024-12-01")

    assert "Error occurred during prediction dataset retrieval" in str(exc_info.value)


def test_fetch_prediction_dataset_without_date(mock_snowflake_session):
    """prediction_dateなしで推論用データセット取得を実行した場合のテスト"""
    with pytest.raises(RuntimeError) as exc_info:
        fetch_prediction_dataset(mock_snowflake_session, prediction_date=None)

    assert "prediction_date is required for inference" in str(exc_info.value)


def test_fetch_empty_training_dataset(mocker):
    """空の学習用データセットが返された場合のテスト"""
    # 空のデータフレームを返すモックセッション
    empty_session = mocker.Mock()
    empty_result = mocker.Mock()
    empty_result.to_pandas.return_value = pd.DataFrame()
    empty_session.sql.return_value = empty_result

    # エラーが発生することを確認
    with pytest.raises(RuntimeError) as exc_info:
        fetch_training_dataset(empty_session)

    assert "Error occurred during training dataset retrieval" in str(exc_info.value)


def test_fetch_empty_prediction_dataset(mocker):
    """空の推論用データセットが返された場合のテスト"""
    # 空のデータフレームを返すモックセッション
    empty_session = mocker.Mock()
    empty_result = mocker.Mock()
    empty_result.to_pandas.return_value = pd.DataFrame()
    empty_session.sql.return_value = empty_result

    # エラーが発生することを確認
    with pytest.raises(RuntimeError) as exc_info:
        fetch_prediction_dataset(empty_session, prediction_date="2024-12-01")

    assert "Error occurred during prediction dataset retrieval" in str(exc_info.value)


def test_fetch_test_dataset(mock_snowflake_session, mocker):
    """テスト用データセット取得のテスト"""
    # モデルバージョンのモック作成
    mock_model_version = mocker.Mock()
    mock_model_version.version_name = "V_250130_121116"  # 2024年3月15日のモデル

    # 実行
    df = fetch_test_dataset(mock_snowflake_session, mock_model_version)

    # アサーション
    assert df is not None
    assert isinstance(df, pd.DataFrame)

    # SQLクエリにテスト用の日付条件が含まれていることを確認
    sql_query = mock_snowflake_session.sql.call_args[0][0]
    expected_start_date = "2025-01-31"  # モデル作成日の翌日
    expected_end_date = "2025-02-13"    # モデル作成日から14日後
    assert f"BETWEEN '{expected_start_date}' AND '{expected_end_date}'" in sql_query


def test_fetch_test_dataset_error(mocker):
    """テスト用データセット取得時のエラーハンドリングテスト"""
    # エラーを発生させるモックセッション
    error_session = mocker.Mock()
    error_session.sql.side_effect = Exception("Database connection failed")
    
    # モデルバージョンのモック
    mock_model_version = mocker.Mock()
    mock_model_version.version_name = "V_250130_121116"

    # エラーが発生することを確認
    with pytest.raises(RuntimeError) as exc_info:
        fetch_test_dataset(error_session, mock_model_version)

    assert "Error occurred during testing dataset retrieval" in str(exc_info.value)


def test_fetch_empty_test_dataset(mocker):
    """空のテスト用データセットが返された場合のテスト"""
    # 空のデータフレームを返すモックセッション
    empty_session = mocker.Mock()
    empty_result = mocker.Mock()
    empty_result.to_pandas.return_value = pd.DataFrame()
    empty_session.sql.return_value = empty_result

    # モデルバージョンのモック
    mock_model_version = mocker.Mock()
    mock_model_version.version_name = "V_250130_121116"

    # エラーが発生することを確認
    with pytest.raises(RuntimeError) as exc_info:
        fetch_test_dataset(empty_session, mock_model_version)

    assert "Error occurred during testing dataset retrieval" in str(exc_info.value)
