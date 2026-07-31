#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QZ / SCADA 检测与打分错误可视化 HTTP 服务

启动：
    python api_server.py

默认监听 0.0.0.0:5000，可通过环境变量覆盖：
    PORT=8080 python api_server.py
    CONFIG_PATH=/workspace/scadaandqz/config.json G_DIR=/workspace/test/scada/g python api_server.py

提供的接口：
    GET  /health
    POST /api/qz/analyze            QZ 检测（支持单张图片路径或目录路径）
    POST /api/qz/analyze/upload     QZ 检测（支持单张/多张图片上传）
    POST /api/scada/analyze         SCADA 检测（支持单张图片路径或目录路径，G文件必填/填目录）
    POST /api/scada/analyze/upload  SCADA 检测（支持单张/多张图片 + G文件上传）
    POST /api/score/qz              QZ 打分与错误可视化
    POST /api/score/scada           SCADA 打分与错误可视化
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from pathlib import Path
from threading import Lock
from flask import Flask, request, jsonify

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api import QZAnalyzer, SCADAAnalyzer, ScoreEngine

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api_server")

app = Flask(__name__)

# ---------- 配置 ----------
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/workspace/scadaandqz/config.json")
G_DIR = os.environ.get("G_DIR", "/workspace/test/scada/g")
TUYUAN_DIR = os.environ.get("TUYUAN_DIR", "/workspace/test/tuyuan")

# 支持的图片后缀
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------- 全局模型实例（懒加载 / 单例复用） ----------
# 模型只在第一次请求或启动预加载时初始化一次，后续所有请求复用，
# 不会每次调用都重新加载。
_qz_analyzer = None
_scada_analyzer = None
_score_engine = None

_qz_lock = Lock()
_scada_lock = Lock()


def get_qz_analyzer() -> QZAnalyzer:
    global _qz_analyzer
    if _qz_analyzer is None:
        with _qz_lock:
            # 双重检查锁定，防止多线程重复初始化
            if _qz_analyzer is None:
                logger.info("[初始化] 正在加载 QZ OCR 模型，配置=%s", CONFIG_PATH)
                _qz_analyzer = QZAnalyzer(CONFIG_PATH)
                logger.info("[初始化] QZAnalyzer 加载完成")
    return _qz_analyzer


def get_scada_analyzer() -> SCADAAnalyzer:
    global _scada_analyzer
    if _scada_analyzer is None:
        with _scada_lock:
            if _scada_analyzer is None:
                logger.info("[初始化] 正在加载 SCADA 模型，配置=%s", CONFIG_PATH)
                _scada_analyzer = SCADAAnalyzer(CONFIG_PATH, G_DIR, TUYUAN_DIR)
                logger.info("[初始化] SCADAAnalyzer 加载完成")
    return _scada_analyzer


def get_score_engine() -> ScoreEngine:
    global _score_engine
    if _score_engine is None:
        _score_engine = ScoreEngine()
    return _score_engine


def preload_models():
    """启动时预加载所有模型，避免第一次请求等待。"""
    logger.info("[预加载] 开始预加载模型...")
    get_qz_analyzer()
    get_scada_analyzer()
    get_score_engine()
    logger.info("[预加载] 全部模型加载完成")


# ---------- 工具函数 ----------
def _load_json_field(value, default=None):
    """从 form/json 字符串中安全解析 dict。"""
    if value is None:
        return default if default is not None else {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default if default is not None else {}


def _collect_image_paths(input_path: Path):
    """收集输入路径中的图片，支持单文件或目录。"""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


# ---------- 健康检查 ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": 200, "service": "qz-scada-score-api"})


# ---------- QZ 接口 ----------
@app.route("/api/qz/analyze", methods=["POST"])
def qz_analyze_path():
    """
    请求体 JSON:
        {
            "image_path": "/workspace/test/qz/img/xxx.jpg",
            "image_path": "/workspace/test/qz/img",       // 也支持目录
            "params": { "idname": "数据点号", "valuenameyc": "数据值", ... }
        }
    返回：单张结果 或 {mode:"batch", count, results:[...]}
    """
    data = request.get_json(force=True, silent=True) or {}
    image_path = data.get("image_dir")
    output_dir = data.get("output_dir")
    params = data.get("params", {})

    if not image_path:
        return jsonify({"status": 400, "error": "缺少 image_dir 参数"}), 400

    input_path = Path(image_path)
    image_paths = _collect_image_paths(input_path)
    if not image_paths:
        return jsonify({"status": 400, "error": f"未找到图片: {image_path}"}), 400

    try:
        with _qz_lock:
            analyzer = get_qz_analyzer()
            if len(image_paths) == 1 and input_path.is_file():
                result = analyzer.analyze(str(image_paths[0]), output_dir, params)
                return jsonify({"status": 200, "mode": "single", "count": len(results), "results": results})
            else:
                results = []
                for img_path in image_paths:
                    try:
                        result = analyzer.analyze(str(img_path), output_dir, params)
                        results.append(result)
                    except Exception as e:
                        logger.exception("QZ 批量分析失败: %s", img_path)
                        results.append({"image_path": str(img_path), "status": "fail", "error": str(e)})
                return jsonify({"status": 200, "mode": "batch", "count": len(results), "results": results})
    except Exception as e:
        logger.exception("QZ 分析失败: %s", image_path)
        return jsonify({"status": 404, "error": str(e)}), 404


@app.route("/api/qz/analyze/upload", methods=["POST"])
def qz_analyze_upload():
    """
    multipart/form-data:
        image: 单个图片文件 或 多个同名图片文件
        params: JSON 字符串（可选）
    返回：单张结果 或 {mode:"batch", count, results:[...]}
    """
    image_files = request.files.getlist("image")
    if not image_files or all(f.filename == "" for f in image_files):
        return jsonify({"status": "fail", "error": "缺少 image 文件"}), 400
    data = request.get_json(force=True, silent=True) or {}
    output_dir = data.get("output_dir")

    # 过滤空文件（某些客户端会附带空项）
    image_files = [f for f in image_files if f and f.filename]
    params = _load_json_field(request.form.get("params"), default={})

    tmp_paths = []
    try:
        for image_file in image_files:
            suffix = Path(image_file.filename or "image.jpg").suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            image_file.save(tmp.name)
            tmp.close()
            tmp_paths.append(Path(tmp.name))

        with _qz_lock:
            analyzer = get_qz_analyzer()
            if len(tmp_paths) == 1:
                result = analyzer.analyze(str(tmp_paths[0]), output_dir, params)
                return jsonify(result)
            else:
                results = []
                for img_path in tmp_paths:
                    try:
                        result = analyzer.analyze(str(img_path), output_dir, params)
                        results.append(result)
                    except Exception as e:
                        logger.exception("QZ 批量上传分析失败: %s", img_path)
                        results.append({"image_path": str(img_path), "status": "fail", "error": str(e)})
                return jsonify({"status": "success", "mode": "batch", "count": len(results), "results": results})
    except Exception as e:
        logger.exception("QZ 上传分析失败")
        return jsonify({"status": "fail", "error": str(e)}), 500
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


# ---------- SCADA 接口 ----------
@app.route("/api/scada/analyze", methods=["POST"])
def scada_analyze_path():
    """
    请求体 JSON:
        单张：
        {
            "image_path": "/workspace/test/scada/img/xxx.png",
            "g_path": "/workspace/test/scada/g/xxx.g"
        }
        目录：
        {
            "image_path": "/workspace/test/scada/img",
            "g_path": "/workspace/test/scada/g"      // 必填，目录；也可省略使用 G_DIR
        }
    返回：单张结果 或 {mode:"batch", count, results:[...]}
    """
    data = request.get_json(force=True, silent=True) or {}
    image_path = data.get("image_dir")
    g_path = data.get("g_dir")
    output_dir = data.get("output_dir")

    if not image_path:
        return jsonify({"status": 400, "error": "缺少 image_dir 参数"}), 400

    input_path = Path(image_path)
    image_paths = _collect_image_paths(input_path)
    if not image_paths:
        return jsonify({"status": 400, "error": f"未找到图片: {image_path}"}), 400

    # 单张模式：g_path 必填且为文件
    if len(image_paths) == 1 and input_path.is_file():
        if not g_path:
            return jsonify({"status": 400, "error": "单张 SCADA 分析需要传入 g_path 参数"}), 400
        try:
            with _scada_lock:
                analyzer = get_scada_analyzer()
                result = analyzer.analyze(str(image_paths[0]), output_dir, g_path)
            return jsonify(result)
        except Exception as e:
            logger.exception("SCADA 路径分析失败: %s", image_path)
            return jsonify({"status": 400, "error": str(e)}), 500

    # 目录模式：g_path 为目录，或使用默认 G_DIR
    g_dir = Path(g_path) if g_path else Path(G_DIR)
    if not g_dir.is_dir():
        return jsonify({"status": 400, "error": f"批量 SCADA 分析需要有效的 G 文件目录: {g_dir}"}), 400

    try:
        with _scada_lock:
            analyzer = get_scada_analyzer()
            results = []
            for img_path in image_paths:
                g_file = g_dir / (img_path.stem + ".g")
                try:
                    if g_file.exists():
                        result = analyzer.analyze(str(img_path), str(output_dir), str(g_file))
                    else:
                        result = {
                            "image_path": str(img_path),
                            "g_file": "",
                            "status": "skip",
                            "error": f"未找到对应 G 文件: {g_file.name}",
                            "data": [],
                        }
                    results.append(result)
                except Exception as e:
                    logger.exception("SCADA 批量分析失败: %s", img_path)
                    results.append({"image_path": str(img_path), "status": "fail", "error": str(e), "data": []})
            return jsonify({"status": 200, "mode": "batch", "count": len(results), "results": results})
    except Exception as e:
        logger.exception("SCADA 批量分析失败: %s", image_path)
        return jsonify({"status": 404, "error": str(e)}), 404


@app.route("/api/scada/analyze/upload", methods=["POST"])
def scada_analyze_upload():
    """
    multipart/form-data:
        image: 单个图片文件 或 多个同名图片文件
        g_file: 单个 G 文件 或 多个同名 G 文件
    多张时按图片 stem 与 G 文件 stem 自动匹配。
    返回：单张结果 或 {mode:"batch", count, results:[...]}
    """
    image_files = request.files.getlist("image")
    g_files = request.files.getlist("g_file")
    data = request.get_json(force=True, silent=True) or {}
    output_dir = data.get("output_dir")

    image_files = [f for f in image_files if f and f.filename]
    g_files = [f for f in g_files if f and f.filename]

    if not image_files:
        return jsonify({"status": "fail", "error": "缺少 image 文件"}), 400
    if not g_files:
        return jsonify({"status": "fail", "error": "缺少 g_file 文件，SCADA 分析需要同时上传 G 文件"}), 400

    tmp_dir = tempfile.mkdtemp()
    try:
        # 保存图片，并记录原始文件 stem
        image_paths = []
        image_stems = []
        for image_file in image_files:
            suffix = Path(image_file.filename or "image.jpg").suffix or ".jpg"
            original_stem = Path(image_file.filename).stem
            save_path = Path(tmp_dir) / f"img_{len(image_paths)}{suffix}"
            image_file.save(str(save_path))
            image_paths.append(save_path)
            image_stems.append(original_stem)

        # 保存 G 文件，并记录原始文件 stem
        g_path_map = {}
        for g_file in g_files:
            suffix = Path(g_file.filename or "image.g").suffix or ".g"
            original_stem = Path(g_file.filename).stem
            save_path = Path(tmp_dir) / f"g_{len(g_path_map)}{suffix}"
            g_file.save(str(save_path))
            g_path_map[original_stem] = save_path

        with _scada_lock:
            analyzer = get_scada_analyzer()
            if len(image_paths) == 1 and len(g_files) == 1:
                # 单张模式：直接用该图片对应的 G 文件
                result = analyzer.analyze(str(image_paths[0]), output_dir, str(g_path_map[image_stems[0]]))
                return jsonify(result)
            else:
                results = []
                for img_path, img_stem in zip(image_paths, image_stems):
                    # 按原始 stem 匹配 G 文件
                    g_file = g_path_map.get(img_stem)
                    try:
                        if g_file and g_file.exists():
                            result = analyzer.analyze(str(img_path), output_dir, str(g_file))
                        else:
                            result = {
                                "image_path": str(img_path),
                                "g_file": "",
                                "status": "skip",
                                "error": f"未找到对应 G 文件: {img_stem}.g",
                                "data": [],
                            }
                        results.append(result)
                    except Exception as e:
                        logger.exception("SCADA 批量上传分析失败: %s", img_path)
                        results.append({"image_path": str(img_path), "status": "fail", "error": str(e), "data": []})
                return jsonify({"status": "success", "mode": "batch", "count": len(results), "results": results})
    except Exception as e:
        logger.exception("SCADA 上传分析失败")
        return jsonify({"status": "fail", "error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------- 打分与错误可视化接口 ----------
@app.route("/api/score/qz", methods=["POST"])
def score_qz():
    """
    请求体 JSON:
        {
            "ref_dir": "/workspace/test/daan/qz",
            "pred_dir": "/workspace/test/qz/result",
            "output_dir": "/workspace/test/qz/score_http",
            "image_dir": "/workspace/test/qz/img"
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    ref_dir = data.get("ref_dir")
    pred_dir = data.get("pred_dir")
    output_dir = data.get("output_dir")
    image_dir = data.get("image_dir")

    if not ref_dir or not pred_dir:
        return jsonify({"status": 400, "error": "缺少 ref_dir 或 pred_dir 参数"}), 400

    try:
        engine = get_score_engine()
        report = engine.score_qz(ref_dir, pred_dir, output_dir, image_dir)
        return jsonify({"status": 200, "report": report})
    except Exception as e:
        logger.exception("QZ 打分失败")
        return jsonify({"status": 404, "error": str(e)}), 500


@app.route("/api/score/scada", methods=["POST"])
def score_scada():
    """
    请求体 JSON:
        {
            "ref_dir": "/workspace/test/daan/scada",
            "pred_dir": "/workspace/test/scada/result",
            "output_dir": "/workspace/test/scada/score_http",
            "image_dir": "/workspace/test/scada/img"
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    ref_dir = data.get("ref_dir")
    pred_dir = data.get("pred_dir")
    output_dir = data.get("output_dir")
    image_dir = data.get("image_dir")

    if not ref_dir or not pred_dir:
        return jsonify({"status": 400, "error": "缺少 ref_dir 或 pred_dir 参数"}), 400

    try:
        engine = get_score_engine()
        report = engine.score_scada(ref_dir, pred_dir, output_dir, image_dir)
        return jsonify({"status": 200, "report": report})
    except Exception as e:
        logger.exception("SCADA 打分失败")
        return jsonify({"status": 404, "error": str(e)}), 500


# ---------- 启动 ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("启动 QZ/SCADA/Score HTTP API，监听 0.0.0.0:%d", port)
    logger.info("CONFIG_PATH=%s, G_DIR=%s, TUYUAN_DIR=%s", CONFIG_PATH, G_DIR, TUYUAN_DIR)

    # 若设置 PRELOAD_MODELS=1，启动时即加载模型，避免第一次请求长时间等待
    # if os.environ.get("PRELOAD_MODELS", "0") in ("1", "true", "True", "TRUE"):
    preload_models()

    app.run(host="0.0.0.0", port=port, threaded=True)
