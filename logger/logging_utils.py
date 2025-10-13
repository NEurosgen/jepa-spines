# logging_utils.py
import logging, os
from logging.handlers import RotatingFileHandler

def setup_logger(name="graphjepa", log_dir="logs", filename="train.log", level=logging.DEBUG):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # чтобы не дублировалось

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # консоль
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # файл с ротацией
    fh = RotatingFileHandler(os.path.join(log_dir, filename), maxBytes=5_000_000, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(fmt)

    # добавить хендлеры 1 раз
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger
