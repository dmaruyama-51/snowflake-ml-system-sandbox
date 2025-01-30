from snowflake.snowpark.session import Session
import logging 
from src.utils.snowflake import upload_dataframe_to_snowflake

logger = logging.getLogger(__name__)

def create_ml_dataset(
    session: Session, 
    target_date: str,
    database_name: str,
    schema_name: str,
    table_name: str = "dataset",
    source_table_name: str = "online_shoppers_intention",
    ) -> None:
    """
    prepare_online_shoppers_data関数で作成されたデータテーブルをsourceとして、
    指定された日付までのデータを取得しSnowflakeにロード

    Args:
        session (Session): Snowflakeセッション
        target_date (str): 対象日付（YYYY-MM-DD）
        database_name (str): データベース名
        schema_name (str): スキーマ名
        table_name (str): テーブル名
        source_table_name (str): ソーステーブル名
    """

    try:
        logger.info(f"Starting dataset generation. Target date: {target_date}")
        gen_query = f"""
        create or replace table {database_name}.{schema_name}.{table_name} as
        select
            UID,
            SESSION_DATE, 
            cast(REVENUE as integer) as REVENUE,
            ADMINISTRATIVE,
            ADMINISTRATIVE_DURATION,
            INFORMATIONAL,
            INFORMATIONAL_DURATION,
            PRODUCTRELATED,
            PRODUCTRELATED_DURATION,
            BOUNCERATES,
            EXITRATES,
            PAGEVALUES,
            SPECIALDAY,
            OPERATINGSYSTEMS,
            BROWSER,
            REGION,
            TRAFFICTYPE,
            VISITORTYPE,
            cast(WEEKEND as integer) as WEEKEND
        from {database_name}.{schema_name}.{source_table_name}
        where session_date <= '{target_date}'
        """
        session.sql(gen_query).collect()
        logger.info(f"Dataset generation completed: {database_name}.{schema_name}.{table_name}")

    except Exception as e:
        logger.error(f"Error occurred during dataset generation: {str(e)}")
        raise

def update_ml_dataset(
    session: Session, 
    target_date: str,
    database_name: str,
    schema_name: str,
    table_name: str = "dataset",
    source_table_name: str = "online_shoppers_intention",
    ) -> None:
    """
    prepare_online_shoppers_data関数で作成されたデータテーブルをsourceとして、
    指定された日付までのデータを取得しSnowflakeにロード

    Args:
        session (Session): Snowflakeセッション
        target_date (str): 対象日付（YYYY-MM-DD）
        database_name (str): データベース名
        schema_name (str): スキーマ名
        table_name (str): テーブル名
        source_table_name (str): ソーステーブル名
    """

    try:
        logger.info(f"Starting dataset update. Target date: {target_date}")
        logger.info(f"Source table: {database_name}.{schema_name}.{source_table_name}")
        
        dataset_query = f"""
            select
                UID,
                SESSION_DATE, 
                cast(REVENUE as integer) as REVENUE,
                ADMINISTRATIVE,
                ADMINISTRATIVE_DURATION,
                INFORMATIONAL,
                INFORMATIONAL_DURATION,
                PRODUCTRELATED,
                PRODUCTRELATED_DURATION,
                BOUNCERATES,
                EXITRATES,
                PAGEVALUES,
                SPECIALDAY,
                OPERATINGSYSTEMS,
                BROWSER,
                REGION,
                TRAFFICTYPE,
                VISITORTYPE,
                cast(WEEKEND as integer) as WEEKEND
            from 
                {database_name}.{schema_name}.{source_table_name}
            where
                session_date = '{target_date}'
        """
        logger.debug(f"Executing query: {dataset_query}")
        append_df = session.sql(dataset_query).to_pandas()
        logger.info(f"Retrieved {len(append_df)} records")

        if len(append_df) > 0:
            logger.info(f"Appending data to table {database_name}.{schema_name}.{table_name}")
            upload_dataframe_to_snowflake(
                session=session,
                df=append_df,
                database_name=database_name,
                schema_name=schema_name,
                table_name=table_name,
                mode="append",
            )
            logger.info("Data append completed successfully")
        else:
            logger.warning(f"No data found for target date: {target_date}")

    except Exception as e:
        logger.error(f"Error occurred during dataset update: {str(e)}")
        raise