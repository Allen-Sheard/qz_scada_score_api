#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QZ / SCADA 检测与打分 异步 HTTP 服务

基于 Flask + ThreadPoolExecutor，适合处理耗时较长的单张/批量分析任务。

启动：
    python api_server_async.py

环境变量：
    PORT=5000                  服务端口
    CONFIG_PATH=...            模型配置
    G_DIR=...                  SCADA G 文件目录
    TUYUAN_DIR=...             SCADA 图元目录
    WORKERS=4                  后台分析线程数（默认 4）
    PRELOAD_MODELS=1           启动时预加载模型

接口：
    POST /api/async/qz/analyze
    POST /api/async/qz/analyze/upload
    POST /api/async/scada/analyze
    POST /api/async/scada/analyze/upload
    POST /api/async/score/qz
    POST /api/async/score/scada
    GET  /api/async/task/<task_id>
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import uuid
import time
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
logger = logging.getLogger("api_server_async")

app = Flask(__name__)

# ---------- 配置 ----------
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/workspace/scadaandqz/config.json")
G_DIR = os.environ.get("G_DIR", "/workspace/test/scada/g")
TUYUAN_DIR = os.environ.get("TUYUAN_DIR", "/workspace/test/tuyuan")
WORKERS = int(os.environ.get("WORKERS", "4"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------- 全局模型实例 ----------
_qz_analyzer = None
_scada_analyzer = None
_score_engine = None

_qz_lock = Lock()
_scada_lock = Lock()
_init_lock = Lock()

# ---------- 任务线程池与状态 ----------
executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="analyzer")
tasks = {}  # task_id -> dict
tasks_lock = Lock()


def get_qz_analyzer() -> QZAnalyzer:
    global _qz_analyzer
    if _qz_analyzer is None:
        with _init_lock:
            if _qz_analyzer is None:
                logger.info("[初始化] 加载 QZ OCR 模型...")
                _qz_analyzer = QZAnalyzer(CONFIG_PATH)
                logger.info("[初始化] QZAnalyzer 加载完成")
    return _qz_analyzer


def get_scada_analyzer() -> SCADAAnalyzer:
    global _scada_analyzer
    if _scada_analyzer is None:
        with _init_lock:
            if _scada_analyzer is None:
                logger.info("[初始化] 加载 SCADA 模型...")
                _scada_analyzer = SCADAAnalyzer(CONFIG_PATH, G_DIR, TUYUAN_DIR)
                logger.info("[初始化] SCADAAnalyzer 加载完成")
    return _scada_analyzer


def get_score_engine() -> ScoreEngine:
    global _score_engine
    if _score_engine is None:
        with _init_lock:
            if _score_engine is None:
                _score_engine = ScoreEngine()
    return _score_engine


def preload_models():
    logger.info("[预加载] 开始预加载模型...")
    get_qz_analyzer()
    get_scada_analyzer()
    get_score_engine()
    logger.info("[预加载] 全部模型加载完成")


# ---------- 任务管理 ----------
def create_task() -> str:
    task_id = str(uuid.uuid4())
    with tasks_lock:
        tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "finished_at": None,
            "progress": None,
        }
    return task_id


def update_task(task_id, status=None, result=None, error=None, progress=None):
    with tasks_lock:
        if task_id not in tasks:
            return
        task = tasks[task_id]
        if status:
            task["status"] = status
        if result is not None:
            task["result"] = result
        if error is not None:
            task["error"] = error
        if progress is not None:
            task["progress"] = progress
        if status in ("success", "fail"):
            task["finished_at"] = datetime.now().isoformat()


def _on_task_done(task_id, tmp_dirs=None):
    """任务完成后的回调：清理临时目录。"""
    def callback(future):
        try:
            result = future.result()
            update_task(task_id, status="success", result=result)
            logger.info("[任务完成] %s", task_id)
        except Exception as e:
            logger.exception("[任务失败] %s", task_id)
            update_task(task_id, status=404, error=str(e))
        finally:
            if tmp_dirs:
                for d in tmp_dirs:
                    try:
                        shutil.rmtree(d, ignore_errors=True)
                    except Exception:
                        pass
    return callback


# ---------- 工具函数 ----------
def _load_json_field(value, default=None):
    if value is None:
        return default if default is not None else {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default if default is not None else {}


def _collect_image_paths(input_path: Path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


# ---------- 实际执行任务 ----------
def _run_qz_analyze(task_id, image_path, params):
    update_task(task_id, status="running")
    input_path = Path(image_path)
    image_paths = _collect_image_paths(input_path)
    if not image_paths:
        raise ValueError(f"未找到图片: {image_path}")

    with _qz_lock:
        analyzer = get_qz_analyzer()
        if len(image_paths) == 1 and input_path.is_file():
            result = analyzer.analyze(str(image_paths[0]), params)
            data = result.get("data", {})
            points = len(data.get("qz_yc", [])) + len(data.get("qz_yx", []))
            errors = 0 if result.get("status") == "success" else 1
            update_task(task_id, progress={
                "current": image_paths[0].name,
                "points": points,
                "errors": errors,
                "progress": "1/1",
            })
            return result
        else:
            results = []
            total_errors = 0
            for idx, img_path in enumerate(image_paths, 1):
                try:
                    result = analyzer.analyze(str(img_path), params)
                    results.append(result)
                    data = result.get("data", {})
                    points = len(data.get("qz_yc", [])) + len(data.get("qz_yx", []))
                    is_error = result.get("status") != "success"
                    if is_error:
                        total_errors += 1
                    update_task(task_id, progress={
                        "current": img_path.name,
                        "points": points,
                        "errors": total_errors,
                        "progress": f"{idx}/{len(image_paths)}",
                    })
                except Exception as e:
                    logger.exception("QZ 批量分析失败: %s", img_path)
                    total_errors += 1
                    results.append({"image_path": str(img_path), "status": "fail", "error": str(e)})
                    update_task(task_id, progress={
                        "current": img_path.name,
                        "points": 0,
                        "errors": total_errors,
                        "progress": f"{idx}/{len(image_paths)}",
                    })
            return {"status": "success", "mode": "batch", "count": len(results), "results": results}


def _run_scada_analyze(task_id, image_path, g_path):
    update_task(task_id, status="running")
    input_path = Path(image_path)
    image_paths = _collect_image_paths(input_path)
    if not image_paths:
        raise ValueError(f"未找到图片: {image_path}")

    if len(image_paths) == 1 and input_path.is_file():
        if not g_path:
            raise ValueError("单张 SCADA 分析需要传入 g_path")
        with _scada_lock:
            analyzer = get_scada_analyzer()
            result = analyzer.analyze(str(image_paths[0]), g_path)
            points = len(result.get("data", []))
            errors = 0 if result.get("status") == "success" else 1
            update_task(task_id, progress={
                "current": image_paths[0].name,
                "points": points,
                "errors": errors,
                "progress": "1/1",
            })
            return result

    g_dir = Path(g_path) if g_path else Path(G_DIR)
    if not g_dir.is_dir():
        raise ValueError(f"批量 SCADA 分析需要有效的 G 文件目录: {g_dir}")

    with _scada_lock:
        analyzer = get_scada_analyzer()
        results = []
        total_errors = 0
        for idx, img_path in enumerate(image_paths, 1):
            g_file = g_dir / (img_path.stem + ".g")
            try:
                if g_file.exists():
                    result = analyzer.analyze(str(img_path), str(g_file))
                else:
                    result = {
                        "image_path": str(img_path),
                        "g_file": "",
                        "status": "skip",
                        "error": f"未找到对应 G 文件: {g_file.name}",
                        "data": [],
                    }
                results.append(result)
                points = len(result.get("data", []))
                is_error = result.get("status") != "success"
                if is_error:
                    total_errors += 1
                update_task(task_id, progress={
                    "current": img_path.name,
                    "points": points,
                    "errors": total_errors,
                    "progress": f"{idx}/{len(image_paths)}",
                })
            except Exception as e:
                logger.exception("SCADA 批量分析失败: %s", img_path)
                total_errors += 1
                results.append({"image_path": str(img_path), "status": "fail", "error": str(e), "data": []})
                update_task(task_id, progress={
                    "current": img_path.name,
                    "points": 0,
                    "errors": total_errors,
                    "progress": f"{idx}/{len(image_paths)}",
                })
        return {"status": "success", "mode": "batch", "count": len(results), "results": results}


def _run_score(task_id, mode, ref_dir, pred_dir, output_dir, image_dir):
    update_task(task_id, status="running")
    engine = get_score_engine()
    if mode == "qz":
        report = engine.score_qz(ref_dir, pred_dir, output_dir, image_dir)
    else:
        report = engine.score_scada(ref_dir, pred_dir, output_dir, image_dir)

    summary = report.get("summary", {})
    update_task(task_id, progress={
        "current": "score_summary",
        "points": summary.get("total_ref_points", 0),
        "errors": summary.get("error_files_count", 0),
        "progress": f"{summary.get('total_files', 0)}/{summary.get('total_files', 0)}",
    })
    return {"status": "success", "report": report}


def _run_qz_upload(task_id, tmp_paths, params):
    update_task(task_id, status="running")
    with _qz_lock:
        analyzer = get_qz_analyzer()
        if len(tmp_paths) == 1:
            result = analyzer.analyze(str(tmp_paths[0]), params)
            data = result.get("data", {})
            points = len(data.get("qz_yc", [])) + len(data.get("qz_yx", []))
            errors = 0 if result.get("status") == "success" else 1
            update_task(task_id, progress={
                "current": tmp_paths[0].name,
                "points": points,
                "errors": errors,
                "progress": "1/1",
            })
            return result
        else:
            results = []
            total_errors = 0
            for idx, img_path in enumerate(tmp_paths, 1):
                try:
                    result = analyzer.analyze(str(img_path), params)
                    results.append(result)
                    data = result.get("data", {})
                    points = len(data.get("qz_yc", [])) + len(data.get("qz_yx", []))
                    is_error = result.get("status") != "success"
                    if is_error:
                        total_errors += 1
                    update_task(task_id, progress={
                        "current": img_path.name,
                        "points": points,
                        "errors": total_errors,
                        "progress": f"{idx}/{len(tmp_paths)}",
                    })
                except Exception as e:
                    logger.exception("QZ 批量上传分析失败: %s", img_path)
                    total_errors += 1
                    results.append({"image_path": str(img_path), "status": "fail", "error": str(e)})
                    update_task(task_id, progress={
                        "current": img_path.name,
                        "points": 0,
                        "errors": total_errors,
                        "progress": f"{idx}/{len(tmp_paths)}",
                    })
            return {"status": "success", "mode": "batch", "count": len(results), "results": results}


def _run_scada_upload(task_id, image_paths, image_stems, g_path_map):
    update_task(task_id, status="running")
    with _scada_lock:
        analyzer = get_scada_analyzer()
        if len(image_paths) == 1:
            result = analyzer.analyze(str(image_paths[0]), str(g_path_map[image_stems[0]]))
            points = len(result.get("data", []))
            errors = 0 if result.get("status") == "success" else 1
            update_task(task_id, progress={
                "current": image_paths[0].name,
                "points": points,
                "errors": errors,
                "progress": "1/1",
            })
            return result
        else:
            results = []
            total_errors = 0
            for idx, (img_path, img_stem) in enumerate(zip(image_paths, image_stems), 1):
                g_file = g_path_map.get(img_stem)
                try:
                    if g_file and g_file.exists():
                        result = analyzer.analyze(str(img_path), str(g_file))
                    else:
                        result = {
                            "image_path": str(img_path),
                            "g_file": "",
                            "status": "skip",
                            "error": f"未找到对应 G 文件: {img_stem}.g",
                            "data": [],
                        }
                    results.append(result)
                    points = len(result.get("data", []))
                    is_error = result.get("status") != "success"
                    if is_error:
                        total_errors += 1
                    update_task(task_id, progress={
                        "current": img_path.name,
                        "points": points,
                        "errors": total_errors,
                        "progress": f"{idx}/{len(image_paths)}",
                    })
                except Exception as e:
                    logger.exception("SCADA 批量上传分析失败: %s", img_path)
                    total_errors += 1
                    results.append({"image_path": str(img_path), "status": "fail", "error": str(e), "data": []})
                    update_task(task_id, progress={
                        "current": img_path.name,
                        "points": 0,
                        "errors": total_errors,
                        "progress": f"{idx}/{len(image_paths)}",
                    })
            return {"status": "success", "mode": "batch", "count": len(results), "results": results}


# ---------- HTTP 接口 ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "success", "service": "qz-scada-score-async-api"})


@app.route("/api/async/task/<task_id>", methods=["GET"])
def get_task(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "fail", "error": "任务不存在"}), 404

    resp = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress"),
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
    }
    if task["status"] == "success":
        resp["result"] = task["result"]
    elif task["status"] == "fail":
        resp["error"] = task["error"]
    return jsonify(resp)


@app.route("/api/async/qz/analyze", methods=["POST"])
def async_qz_analyze_path():
    data = request.get_json(force=True, silent=True) or {}
    image_path = data.get("image_path")
    params = data.get("params", {})
    if not image_path:
        return jsonify({"status": "fail", "error": "缺少 image_path 参数"}), 400

    task_id = create_task()
    future = executor.submit(_run_qz_analyze, task_id, image_path, params)
    future.add_done_callback(_on_task_done(task_id))
    return jsonify({"status": "success", "task_id": task_id, "message": "任务已提交"})


@app.route("/api/async/qz/analyze/upload", methods=["POST"])
def async_qz_analyze_upload():
    image_files = request.files.getlist("image")
    image_files = [f for f in image_files if f and f.filename]
    if not image_files:
        return jsonify({"status": "fail", "error": "缺少 image 文件"}), 400

    params = _load_json_field(request.form.get("params"), default={})

    tmp_paths = []
    for image_file in image_files:
        suffix = Path(image_file.filename or "image.jpg").suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        image_file.save(tmp.name)
        tmp.close()
        tmp_paths.append(Path(tmp.name))

    task_id = create_task()
    future = executor.submit(_run_qz_upload, task_id, tmp_paths, params)
    future.add_done_callback(_on_task_done(task_id))
    return jsonify({"status": "success", "task_id": task_id, "message": "任务已提交"})


@app.route("/api/async/scada/analyze", methods=["POST"])
def async_scada_analyze_path():
    data = request.get_json(force=True, silent=True) or {}
    image_path = data.get("image_path")
    g_path = data.get("g_path")
    if not image_path:
        return jsonify({"status": "fail", "error": "缺少 image_path 参数"}), 400

    task_id = create_task()
    future = executor.submit(_run_scada_analyze, task_id, image_path, g_path)
    future.add_done_callback(_on_task_done(task_id))
    return jsonify({"status": "success", "task_id": task_id, "message": "任务已提交"})


@app.route("/api/async/scada/analyze/upload", methods=["POST"])
def async_scada_analyze_upload():
    image_files = request.files.getlist("image")
    g_files = request.files.getlist("g_file")
    image_files = [f for f in image_files if f and f.filename]
    g_files = [f for f in g_files if f and f.filename]

    if not image_files:
        return jsonify({"status": "fail", "error": "缺少 image 文件"}), 400
    if not g_files:
        return jsonify({"status": "fail", "error": "缺少 g_file 文件"}), 400

    tmp_dir = tempfile.mkdtemp()
    try:
        image_paths = []
        image_stems = []
        for image_file in image_files:
            suffix = Path(image_file.filename or "image.jpg").suffix or ".jpg"
            original_stem = Path(image_file.filename).stem
            save_path = Path(tmp_dir) / f"img_{len(image_paths)}{suffix}"
            image_file.save(str(save_path))
            image_paths.append(save_path)
            image_stems.append(original_stem)

        g_path_map = {}
        for g_file in g_files:
            suffix = Path(g_file.filename or "image.g").suffix or ".g"
            original_stem = Path(g_file.filename).stem
            save_path = Path(tmp_dir) / f"g_{len(g_path_map)}{suffix}"
            g_file.save(str(save_path))
            g_path_map[original_stem] = save_path
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"status": "fail", "error": f"保存上传文件失败: {e}"}), 500

    task_id = create_task()
    future = executor.submit(_run_scada_upload, task_id, image_paths, image_stems, g_path_map)
    future.add_done_callback(_on_task_done(task_id, tmp_dirs=[tmp_dir]))
    return jsonify({"status": "success", "task_id": task_id, "message": "任务已提交"})


@app.route("/api/async/score/qz", methods=["POST"])
def async_score_qz():
    data = request.get_json(force=True, silent=True) or {}
    ref_dir = data.get("ref_dir")
    pred_dir = data.get("pred_dir")
    output_dir = data.get("output_dir")
    image_dir = data.get("image_dir")
    if not ref_dir or not pred_dir:
        return jsonify({"status": "fail", "error": "缺少 ref_dir 或 pred_dir 参数"}), 400

    task_id = create_task()
    future = executor.submit(_run_score, task_id, "qz", ref_dir, pred_dir, output_dir, image_dir)
    future.add_done_callback(_on_task_done(task_id))
    return jsonify({"status": "success", "task_id": task_id, "message": "任务已提交"})


@app.route("/api/async/score/scada", methods=["POST"])
def async_score_scada():
    data = request.get_json(force=True, silent=True) or {}
    ref_dir = data.get("ref_dir")
    pred_dir = data.get("pred_dir")
    output_dir = data.get("output_dir")
    image_dir = data.get("image_dir")
    if not ref_dir or not pred_dir:
        return jsonify({"status": "fail", "error": "缺少 ref_dir 或 pred_dir 参数"}), 400

    task_id = create_task()
    future = executor.submit(_run_score, task_id, "scada", ref_dir, pred_dir, output_dir, image_dir)
    future.add_done_callback(_on_task_done(task_id))
    return jsonify({"status": "success", "task_id": task_id, "message": "任务已提交"})


# ---------- 启动 ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("启动异步 QZ/SCADA/Score HTTP API，监听 0.0.0.0:%d", port)
    logger.info("CONFIG_PATH=%s, G_DIR=%s, TUYUAN_DIR=%s, WORKERS=%d", CONFIG_PATH, G_DIR, TUYUAN_DIR, WORKERS)

    # if os.environ.get("PRELOAD_MODELS", "0") in ("1", "true", "True", "TRUE"):
    preload_models()

    app.run(host="0.0.0.0", port=port, threaded=True)
