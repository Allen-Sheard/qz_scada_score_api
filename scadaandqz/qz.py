import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ========== 第一步：环境变量级禁用PyTorch CuDNN（代码最顶部） ==========
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_DOWNLOAD_ENABLED"] = "False"
os.environ["PADDLE_PDX_DISABLE_FONT_DOWNLOAD"] = "True"
os.environ["PADDLE_PDX_DISABLE_DICT_DOWNLOAD"] = "True"
# 新增：彻底关闭PaddleX/PaddleHub模型仓库探测
os.environ["PADDLE_MODEL_HUB_DISABLE"] = "True"
os.environ["PADDLE_HUB_DISABLE"] = "True"
os.environ["PPX_MODEL_HUB_DISABLE"] = "True"

import warnings
warnings.filterwarnings("ignore", message=".*No model host.*")

# 1. 全局禁用PyTorch的CuDNN（优先级最高）
os.environ["TORCH_BACKENDS_CUDNN_ENABLED"] = "False"
os.environ["TORCH_BACKENDS_CUDNN_AUTOTUNE_ENABLED"] = "False"
os.environ["TORCH_BACKENDS_CUDNN_BENCHMARK"] = "False"

# ========== 第二步：代码级确认禁用PyTorch CuDNN ==========
import torch

# 强制关闭CuDNN（二次确认）
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = False

# 验证PyTorch CuDNN状态
print("===== PyTorch 配置 =====")
print("PyTorch CuDNN启用状态：", torch.backends.cudnn.enabled)
print("PyTorch CuDNN版本（禁用后为None）：", torch.backends.cudnn.version() if torch.backends.cudnn.enabled else "None")
print("PyTorch CUDA可用：", torch.cuda.is_available())

# ========== 第三步：配置PaddlePaddle（正确的GPU调用方式） ==========
import paddle

# Paddle指定GPU设备（核心修复：替代.cuda()）
paddle.device.set_device("gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu")

# 验证Paddle配置
print("\n===== Paddle 配置 =====")
print("Paddle设备：", paddle.device.get_device())
print("Paddle CUDA可用：", paddle.device.is_compiled_with_cuda())



import json
from flask import Flask, request, jsonify

# ======================== 导入你的模块 ========================
from base_recognize import BaseRecognitionManager, setup_logger
from table import OCRRecognizer

# ======================== 全局初始化 ========================
app = Flask(__name__)

# 加载配置
CONFIG_PATH = "./config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    GLOBAL_CONFIG = json.load(f)

logger = setup_logger("MainServer")

# 初始化识别管理器
logger.info("初始化识别管理器...")
recognition_manager = BaseRecognitionManager(
    recognizer_class=OCRRecognizer
)

# 启动工作线程
recognition_manager.start_worker(model_paths=GLOBAL_CONFIG["table_model_paths"])
logger.info("识别管理器初始化完成")

# ======================== Flask 接口 ========================

@app.route("/api/start_recognition", methods=["POST"])
def start_recognition():
    try:
        req = request.get_json()
        camera_id = req["camera_id"]
        params = req.get("params", {})

        recognition_manager.send_start_cmd(
            camera_id=camera_id,
            params=params
        )

        return jsonify({
            "code": 200,
            "msg": "已启动识别",
            "data": {"camera_id": camera_id}
        })

    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": {}})


@app.route("/api/stop_recognition", methods=["POST"])
def stop_recognition():
    try:
        recognition_manager.send_stop_cmd()
        return jsonify({"code": 200, "msg": "已停止识别", "data": {}})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": {}})


@app.route("/api/get_status", methods=["GET"])
def get_status():
    status = {
        "worker_running": recognition_manager.worker_running,
        "is_recognizing": recognition_manager.is_recognizing,
        "current_camera_id": recognition_manager.current_camera_id,
        "model_init_done": recognition_manager.model_init_done,
    }
    return jsonify({"code": 200, "msg": "ok", "data": status})


# ======================== 核心：获取最新识别结果（线程安全） ========================
@app.route("/api/get_latest_result", methods=["GET"])
def get_latest_result():
    try:
        with recognition_manager.result_lock:
            data = recognition_manager.latest_recognition_result.copy()

        return jsonify({
            "code": 200,
            "msg": "success",
            "data": data
        })
    except Exception as e:
        return jsonify({
            "code": 500,
            "msg": str(e),
            "data": {}
        })
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

# ======================== 启动服务 ========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=9000,
        threaded=True,
        debug=False,
        use_reloader=False
    )