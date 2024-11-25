import logging.config
from pathlib import Path

# ログファイルを保存するディレクトリを設定
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    # ログの出力形式を定義
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            # %(asctime)s: タイムスタンプ
            # %(levelname)s: ログレベル（INFO, ERROR等）
            # %(name)s: ロガー名（モジュール名）
            # %(message)s: ログメッセージ
        }
    },
    # ログの出力先を定義
    "handlers": {
        # コンソールへの出力設定
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",  # INFO以上のログレベルを出力
            "formatter": "standard",
        },
        # ファイルへの出力設定
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",  # DEBUG以上のログレベルを出力
            "formatter": "standard",
            "filename": LOG_DIR / "app.log",
            "maxBytes": 10485760,  # ファイルの最大サイズ（10MB）
            "backupCount": 3,  # 保持する過去ログファイルの数
        },
    },
    "loggers": {
        "src": {  # srcパッケージ配下のロガー設定
            "level": "DEBUG",  # DEBUG以上のログレベルを処理
            "handlers": ["console", "file"],
        }
    },
}


def setup_logging():
    """ロギング設定を初期化"""
    logging.config.dictConfig(LOGGING_CONFIG)
