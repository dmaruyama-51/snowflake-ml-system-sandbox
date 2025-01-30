import logging
from datetime import datetime

from src.data.dataset import create_ml_dataset
from src.data.source import prepare_online_shoppers_data
from src.utils.logger import setup_logging
from src.utils.snowflake import create_session

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        setup_logging()

        
        logger.info("Creating Snowflake session")
        session = create_session()
        if session is None:
            raise ValueError("Failed to create Snowflake session")
        logger.info("Snowflake session created successfully")

        prepare_online_shoppers_data(
            session=session,
            database_name="practice",
            schema_name="ml",
            table_name="online_shoppers_intention",
        )

        today = datetime.now().strftime("%Y-%m-%d")
        create_ml_dataset(
            session=session,
            target_date=today,
            database_name="practice",
            schema_name="ml",
            table_name="dataset",
            source_table_name="online_shoppers_intention",
        )
    finally:
        if session:
            logger.info("Closing Snowflake session")
            session.close()
