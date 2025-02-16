import argparse
import logging

from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from src.utils.logger import setup_logging
from src.utils.snowflake import create_session

logger = logging.getLogger(__name__)


def rollback_model(session: Session, version_name: str) -> None:
    """
    指定されたバージョンにモデルをロールバックする

    Args:
        session (Session): Snowflakeセッション
        version_name (str): ロールバック先のバージョン名

    Raises:
        Exception: ロールバック処理中にエラーが発生した場合
    """
    try:
        registry = Registry(session=session)
        model_ref = registry.get_model("random_forest")

        # 指定されたバージョンが存在するか確認
        try:
            target_version = model_ref.version(version_name)
        except Exception as e:
            raise ValueError(f"Specified version {version_name} not found: {str(e)}")

        # 現在のデフォルトバージョンを取得
        current_default = model_ref.default

        logger.info(f"Current default version: {current_default.version_name}")
        logger.info(f"Target rollback version: {version_name}")

        # デフォルトバージョンを更新
        model_ref.default = target_version
        logger.info(f"Default version updated to {version_name}")

    except Exception as e:
        error_msg = f"An error occurred during rollback process: {str(e)}"
        logger.error(error_msg)
        raise


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Model rollback process")
    parser.add_argument(
        "version_name",
        type=str,
        help="Target version name for rollback (e.g., v_250130_121116)",
    )
    args = parser.parse_args()

    try:
        session = create_session()
        if session is None:
            raise RuntimeError("Failed to create Snowflake session")

        rollback_model(session, args.version_name)
        logger.info("Rollback process completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if session:
            session.close()


if __name__ == "__main__":
    main()
