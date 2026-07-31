import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify

# ======================== 导入你的模块 ========================
# 注意：请根据实际文件路径调整导入
from base_recognize import BatchFrameRecognitionManager  # 你的批量识别管理器
from common.scada_algorithm import AlgorithmScheduler                # 你的OCR识别器
from base_recognize import setup_logger                                # 你的日志配置

# ======================== 全局初始化（程序启动时执行）========================
app = Flask(__name__)

# 1. 加载配置文件
CONFIG_PATH = "./config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    GLOBAL_CONFIG = json.load(f)

# 2. 初始化日志
logger = setup_logger("SCADAServer")

# 3. 自定义推送函数（适配主题，全局复用）
# 4. 提前初始化识别管理器（程序启动时只执行一次）
logger.info("开始初始化SCADA识别管理器...")
SCADA_manager = BatchFrameRecognitionManager(
    recognizer_class=AlgorithmScheduler
)

# 5. 启动工作线程（加载模型，仅启动一次）
#recognition_manager.MAX_RECONNECT_COUNT = GLOBAL_CONFIG["max_reconnect_count"]
SCADA_manager.start_worker(model_paths=GLOBAL_CONFIG["scada_model_paths"])
logger.info("SCADA识别管理器初始化完成，工作线程已启动")

# ======================== Flask 接口（仅发送指令）========================
@app.route("/api/start_SCADA", methods=["POST"])
def start_recognition():
    """
    启动/切换表格OCR分析任务（仅发送指令，不初始化管理器）
    请求示例：
    {
        "camera_id": 0,                # 摄像头ID（0为默认摄像头，支持视频文件路径）
        "params": {}# 自定义参数
    }
    """
    try:
        # 1. 解析请求参数
        req_data = request.get_json()
        if not req_data:
            return jsonify({
                "code": 400,
                "msg": "请求体不能为空",
                "data": {}
            }), 400
        
        # 3. 解析参数（使用配置默认值）
        camera_id = req_data["camera_id"]
        params = req_data.get("params", {})
        
        # 4. 仅发送启动/切换指令（核心：不重新初始化，只传动态参数）
        # 指令参数格式：(camera_id, push_url, push_topic, ocr_params)
        SCADA_manager.send_start_cmd(
            camera_id=camera_id,
            params=params
        )
        
        return jsonify({
            "code": 200,
            "msg": "SCADA分析任务指令已发送",
            "data": {
                "camera_id": camera_id,
                "params": params
            }
        }), 200
    
    except Exception as e:
        logger.error(f"发送启动指令异常：{str(e)}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": f"发送指令失败：{str(e)}",
            "data": {}
        }), 500

@app.route("/api/stop_SCADA", methods=["POST"])
def stop_recognition():
    """停止分析任务（仅发送停止指令）"""
    try:
        # 仅发送停止指令，不销毁管理器
        SCADA_manager.send_stop_cmd()
        logger.info("已发送停止识别任务指令")
        
        return jsonify({
            "code": 200,
            "msg": "停止指令已发送，分析任务将停止",
            "data": {}
        }), 200
    
    except Exception as e:
        logger.error(f"发送停止指令异常：{str(e)}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": f"发送停止指令失败：{str(e)}",
            "data": {}
        }), 500


# ======================== 启动服务 ========================
if __name__ == "__main__":
    # 启动Flask服务（线程模式，支持多请求）
    app.run(
        host="0.0.0.0",
        port=9001,
        threaded=True,
        debug=False,
        use_reloader=False  # 禁用重载，避免管理器重复初始化
    )