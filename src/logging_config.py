import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path(os.getenv("LOG_DIR", "/app/logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILENAME = "app.log"

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging() -> None:
    """Konfiguruje root logger: codzienna rotacja pliku + stdout. Idempotentne."""
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    file_handler = TimedRotatingFileHandler(
        LOG_DIR / LOG_FILENAME,
        when="midnight",
        backupCount=0,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    sys.excepthook = _log_uncaught
    _configured = True

    logging.getLogger(__name__).info(
        "Logowanie skonfigurowane: dir=%s level=%s rotation=daily backup=keep_forever",
        LOG_DIR, LOG_LEVEL,
    )


def _log_uncaught(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logging.getLogger("uncaught").critical(
        "Niezłapany wyjątek — proces kończy działanie",
        exc_info=(exc_type, exc_value, exc_tb),
    )
