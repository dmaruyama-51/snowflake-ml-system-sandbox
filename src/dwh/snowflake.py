import json
import logging
from typing import Optional

from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSessionException

logger = logging.getLogger(__name__)
CONNECTION_PARAMETERS_PATH = "connection_parameters.json"


def create_session() -> Optional[Session]:
    """
    snowpark session を作成

    Returns:
        Session: 成功時はSnowparkセッション、失敗時はNone

    Raises:
        SnowparkSessionException: Snowflakeへの接続に失敗した場合
    """
    try:
        logger.info("Snowflakeセッションの作成を開始")
        logger.debug(f"設定ファイルを読み込み: {CONNECTION_PARAMETERS_PATH}")
        with open(CONNECTION_PARAMETERS_PATH) as f:
            connection_parameters = json.load(f)
        logger.info("Snowflakeへの接続を試行")
        session = Session.builder.configs(connection_parameters).create()
        logger.info("Snowflakeセッションの作成に成功")
        return session

    except SnowparkSessionException as e:
        error_msg = f"Snowflakeへの接続に失敗しました: {str(e)}"
        logger.error(error_msg)
        raise SnowparkSessionException(error_msg) from e
