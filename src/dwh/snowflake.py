import json
from typing import Optional

from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSessionException

CONNECTION_PARAMETERS_PATH = "connection_parameters.json"


def create_session() -> Optional[Session]:
    """
    snowpark session を作成

    Returns:
        Session: 成功時はSnowparkセッション、失敗時はNone

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        json.JSONDecodeError: 設定ファイルのJSONが不正な場合
        SnowparkSessionException: Snowflakeへの接続に失敗した場合
    """
    try:
        with open(CONNECTION_PARAMETERS_PATH) as f:
            connection_parameters = json.load(f)

        return Session.builder.configs(connection_parameters).create()

    except SnowparkSessionException as e:
        raise SnowparkSessionException(
            f"Snowflakeへの接続に失敗しました: {str(e)}"
        ) from e
