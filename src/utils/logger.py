import logging.config
from pathlib import Path


# ロァイルハンドラーを条件付きで設定
def get_logging_config():
    """環境に応じたロギング設定を返す"""
    config = {
        "version": 1,
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
            }
        },
        "loggers": {
            "src": {
                "level": "DEBUG",
                "handlers": ["console"],
            }
        },
    }

    # ローカル環境の場合のみファイルハンドラーを追加
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": log_dir / "app.log",
            "maxBytes": 10485760,
            "backupCount": 3,
        }
        config["loggers"]["src"]["handlers"].append("file")
    except OSError:
        # ファイルシステムが読み取り専用の場合はスキップ
        pass

    return config


def setup_logging():
    """ロギング設定を初期化"""
    logging.config.dictConfig(get_logging_config())
