import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス。Noneの場合はデフォルトのパスを使用
    Returns:
        Dict[str, Any]: 設定内容を含む辞書
    """
    if config_path is None:
        root_dir = Path(__file__).parent.parent.parent

        # CI環境またはローカル環境の場合
        if "snowflake_import_directory" not in sys._xoptions:
            config_path = f"{root_dir}/src/config.yml"
        # ストアドプロシージャ内で実行されている場合
        else:  # pragma: no cover
            config_path = os.path.join(
                sys._xoptions["snowflake_import_directory"], "config.yml"
            )

    logger.info(f"Loading config file: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Config file loaded successfully")
        return config

    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to load config file: {str(e)}")
        raise
