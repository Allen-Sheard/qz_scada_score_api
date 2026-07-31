"""
统一配置加载器
==============
使用方式：
    from common.config import get_service_config, PROJECT_ROOT

    cfg = get_service_config("scada")
    port = cfg["http_port"]
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json_config(path: str) -> dict:
    """加载 JSON 配置文件"""
    full_path = PROJECT_ROOT / path
    if not full_path.exists():
        return {}
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_service_config(service_name: str) -> dict:
    """
    获取指定服务的配置
    :param service_name: 服务名，如 scada / qz / mouse / sortandcheck / tuyuan_check / control
    :return: 配置字典
    """
    services = load_json_config("config/services.json")
    return services.get(service_name, {})


def get_log_dir() -> Path:
    """获取统一日志目录"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir
