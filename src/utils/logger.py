import logging
from pathlib import Path
from datetime import datetime


def setup_logger(
    logger_name: str = "service_migration",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    创建并返回一个同时输出到控制台和文件的 logger。
    """

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{logger_name}_{timestamp}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 输出到文件
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 输出到终端
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.propagate = False

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger