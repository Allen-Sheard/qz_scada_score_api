"""
统一日志配置模块
================
使用方式：
    1. 在程序入口（如 main.py）调用一次 setup_logging()
    2. 在各模块顶部用 logger = logging.getLogger(__name__)
    3. 使用 logger.info() / logger.debug() / logger.warning() / logger.error()
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# 日志文件保存目录
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 统一的日志格式
LOG_FORMAT = "[%(asctime)s] [%(levelname)-7s] [%(filename)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 控制台带简短颜色（仅 Windows 也兼容）
LEVEL_COLORS = {
    "DEBUG": "\033[36m",      # 青色
    "INFO": "\033[32m",       # 绿色
    "WARNING": "\033[33m",    # 黄色
    "ERROR": "\033[31m",      # 红色
    "CRITICAL": "\033[35m",   # 紫色
}
RESET_COLOR = "\033[0m"


class ColorFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器"""
    def format(self, record):
        color = LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{RESET_COLOR}"
        return super().format(record)


def setup_logging(
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    log_dir=None,
    log_name="app"
):
    """
    初始化全局日志配置
    :param console_level: 控制台日志级别（默认 INFO）
    :param file_level: 文件日志级别（默认 DEBUG）
    :param log_dir: 日志目录，默认当前目录下的 logs/
    :param log_name: 日志文件名前缀
    """
    log_dir = Path(log_dir) if log_dir else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # 获取根 logger，清除已有 handler（防止重复）
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    # ---- 控制台 Handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = ColorFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ---- 文件 Handler（按天轮转，保留30天）----
    log_file = log_dir / f"{log_name}.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 降低第三方库的日志噪音
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    logging.info(f"日志系统初始化完成 | 控制台≥{logging.getLevelName(console_level)} | 文件≥{logging.getLevelName(file_level)} | 目录={log_dir.absolute()}")


# 兼容：也可以直接导入 logger 使用（非根 logger）
logger = logging.getLogger("app")
