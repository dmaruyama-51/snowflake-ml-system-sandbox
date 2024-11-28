from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

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
        config_path = f"{root_dir}/src/config.yml"
        print(config_path)

    logger.info(f"設定ファイルの読み込みを開始: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("設定ファイルの読み込みが完了しました")
        return config

    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗: {str(e)}")
        raise
