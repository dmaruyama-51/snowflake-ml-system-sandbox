from ucimlrepo import fetch_ucirepo
import pandas as pd
from src.utils.snowflake import upload_dataframe_to_snowflake, create_session
from snowflake.snowpark.session import Session


def prepare_online_shoppers_data(
    session: Session,
    database_name: str,
    schema_name: str,
    table_name: str,
    mode: str = "overwrite",
) -> None:
    """
    Online Shoppers Intention データセットを取得し、Snowflakeにロードする

    Args:
        session (AsyncSession): Snowflakeセッション
        database_name (str): ロード先のデータベース名
        schema_name (str): ロード先のスキーマ名
        table_name (str): ロード先のテーブル名
        mode (str, optional): データ書き込みモード. Defaults to 'overwrite'.

    Raises:
        Exception: データの取得中にエラーが発生した場合
    """
    try:
        # データセットの取得
        dataset = fetch_ucirepo(id=468)
        df: pd.DataFrame = dataset.data.features
        df_target: pd.DataFrame = dataset.data.targets
        df["revenue"] = df_target["Revenue"]

        # MONTHカラムの値を月番号に変換する辞書を作成
        month_to_num = {
            "Jan": "01",
            "Feb": "02",
            "Mar": "03",
            "Apr": "04",
            "May": "05",
            "June": "06",
            "Jul": "07",
            "Aug": "08",
            "Sep": "09",
            "Oct": "10",
            "Nov": "11",
            "Dec": "12",
        }

        # 月しかわからないため、日付は 2024-xx-01 とする
        df["SESSION_DATE"] = pd.to_datetime(
            "2024" + df["Month"].map(month_to_num) + "01"
        )

        # Snowflakeへのアップロード
        upload_dataframe_to_snowflake(
            session=session,
            df=df,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            mode=mode,
        )

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        session = create_session()
        if session is None:
            raise ValueError("Snowflakeセッションの作成に失敗しました")

        prepare_online_shoppers_data(
            session=session,
            database_name="practice",
            schema_name="ml",
            table_name="online_shoppers_intention",
        )
    finally:
        if session:
            session.close()
