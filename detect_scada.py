# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
SCADA 画面批量检测脚本
基于 common.analyse_test.py 直接解析 G 文件，不再依赖 CIME、txt 点表和 map JSON。
输入：图片目录、G 文件目录、图元目录
输出：与 G 文件同名的 LabelMe 格式 JSON 文件
label 格式：点号-值-遥测遥信类型

用法:
    python detect_scada.py \\
        --image-dir /workspace/test/scada/img \\
        --g-dir /workspace/test/scada/g \\
        --tuyuan-dir /workspace/test/tuyuan \\
        --output-dir /workspace/test/scada/result \\
        --config /workspace/scadaandqz/config.json
"""

import os
import re
import argparse
import json
import logging
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from common.scada_algorithm import AlgorithmScheduler
from common.analyse_test import jx_M,M_jsuan,FromGstoMapSGZ,SimpleFileFinder

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detect_scada")

# ---------- cimetypes → 算法类型映射 ----------
CIMETYPE_ALGORITHM_MAP_SH = {
    "cbreaker": "gzp",            # 断路器 → 开关位置
    "disconnector": "gzp",      # 隔离刀闸 → 开关位置
    "grounddisconnector": "gzp",  # 接地刀闸 → 接地状态
    "dollybreaker_kg": "gzp",     # 小车开关位置
    "dollybreaker_sc": "sc",      # 小车开关状态
    "dtext": "dtext",             # 数字/文本识别
    "gzp": "gzp_l",
    "protect": None,              # 保护信号（暂跳过）
}

CIMETYPE_ALGORITHM_MAP_SB = {
    "cbreaker": "kg",            # 断路器 → 开关位置
    "disconnector": "dz",      # 隔离刀闸 → 开关位置
    "grounddisconnector": "jddz",  # 接地刀闸 → 接地状态
    "dollybreaker_kg": "kg",     # 小车开关位置
    "dollybreaker_sc": "sc",      # 小车开关状态
    "dtext": "dtext",             # 数字/文本识别
    "gzp": "gzp",
    "protect": None,              # 保护信号（暂跳过）
}

# ---------- 工具函数 ----------
_G_PIC_SUFFIX = re.compile(r'^(.+?)(\.(bay|fac))?\.pic(\d*)\.g$', re.IGNORECASE)


def find_g_for_image(image_path: Path, g_dir: Path) -> Optional[Path]:
    """
    根据 G 文件名查找对应图片（支持 .jpg/.jpeg/.png/.bmp）。
    支持 G 文件后缀如：
        x.bay.pic.g, x.bay.pic2.g, x.fac.pic.g, x.fac.pic3.g, x.pic.g, x.pic4.g
    对应图片优先保留完全一致的后缀结构：
        x.bay.pic.jpg, x.bay.pic2.jpg, x.fac.pic.jpg, x.fac.pic3.jpg, ...
    """
    name = image_path.name
    match = _G_PIC_SUFFIX.match(name)

    if match:
        base, sub_type_dot, _, version = match.groups()
        sub_type = sub_type_dot or ""
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = g_dir / f"{base}{sub_type}.pic{version}{ext}"
            if candidate.exists():
                return candidate
        # 回退：不带 bay/fac 子类型
        if sub_type:
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = g_dir / f"{base}.pic{version}{ext}"
                if candidate.exists():
                    return candidate

    # 通用回退：直接用 G 文件 stem 匹配图片
    base = image_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = g_dir / (base + ext)
        if candidate.exists():
            return candidate
    return None


def format_yc_value(value: str) -> str:
    """
    - 含小数点但小数位 > 2：截断至两位小数。
    """
    if not value or not isinstance(value, str) or "." not in value:
        return value

    sign = ""
    num = value
    if num.startswith("-"):
        sign = "-"
        num = num[1:]

    integer_part, decimal_part = num.split(".", 1)
    if len(decimal_part) > 2:
        decimal_part = decimal_part[:2]
    return f"{sign}{integer_part}.{decimal_part}"


def analyze_g_node(
    image: np.ndarray,
    node: Dict[str, Any],
    scheduler: AlgorithmScheduler,
    algorithm_map: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """
    基于 analyse_test.py 解析出的单个节点，对其 points 逐点位调用算法识别。
    """
    g_size = node.get("size", [0, 0])
    points = node.get("points", [])
    node_name = node.get("name", "")

    if not g_size or g_size[0] == 0:
        logger.warning("节点 %s 缺少有效 size，跳过", node_name)
        return []

    # 计算 G 文件坐标 → 屏幕坐标的变换矩阵
    M = jx_M(image, g_size)

    detected = []
    img_h, img_w = image.shape[:2]
    for idx, pt in enumerate(points):
        cimeid = pt.get("cimeid", "")
        cimetypes = pt.get("cimetypes", "")
        cimeboxs = pt.get("cimeboxs", [])
        ycyx = pt.get("ycyx", "")
        cimename = pt.get("cimename", "")

        # 跳过不支持的图元类型
        alg_type = algorithm_map.get(cimetypes)
        if alg_type is None:
            continue

        # 坐标变换: G 文件坐标 → 屏幕坐标
        if not cimeboxs or len(cimeboxs) != 4:
            continue
        screen_box = M_jsuan(cimeboxs, M)
        if not screen_box or len(screen_box) != 4:
            continue
        x1, y1, x2, y2 = screen_box

        # 边界裁剪（防止越界）
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # 构建算法输入参数
        input_data = {
            "algorithm_type": alg_type,
            "yxyc_type": ycyx,
            "cime_ID": cimeid,
            "roi": screen_box,
        }

        try:
            result, _ = scheduler.run([image], input_data)
            status = result.get("status", "fail")

            value = ""
            if status == "success":
                res_data = result.get("Res", {})
                if ycyx == "yx" and res_data.get("jxt_yx"):
                    value = str(res_data["jxt_yx"][0].get("value", ""))
                elif ycyx == "yc" and res_data.get("jxt_yc"):
                    raw_value = str(res_data["jxt_yc"][0].get("value", ""))
                    value = format_yc_value(raw_value)

            detected.append({
                "cimeid": cimeid,
                "cimename": cimename,
                "cimetypes": cimetypes,
                "ycyx": ycyx,
                "roi": screen_box,
                "value": value,
                "status": status,
            })

            if idx % 20 == 0:
                logger.info("节点 %s 进度: %d/%d", node_name, idx + 1, len(points))

        except Exception as e:
            logger.warning("点位 %s 识别失败: %s", cimeid, e)

    return detected


def build_labelme_json(
    image_path: str,
    image_size: Tuple[int, int],
    points: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    将识别结果转换为 LabelMe 格式 JSON。
    label 格式：点号-值-遥测遥信类型
    """
    shapes = []
    for pt in points:
        roi = pt.get("roi")
        if not roi or len(roi) != 4:
            continue
        x1, y1, x2, y2 = roi
        cimeid = str(pt.get("cimeid", ""))
        value = str(pt.get("value", ""))
        ycyx = str(pt.get("ycyx", ""))
        label = f"{cimeid}_{value}_{ycyx}"
        shapes.append({
            "label": label,
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
        })

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_size[1],
        "imageWidth": image_size[0],
    }


def process_g_file(
    g_path: Path,
    image_path: Path,
    output_dir: Path,
    file_finder: SimpleFileFinder,
    scheduler: AlgorithmScheduler,
    save_labelme: str
) -> Dict[str, Any]:
    """
    处理单个 G 文件：解析 → 识别 →（可选）保存 LabelMe JSON。
    返回识别结果字典，便于主函数汇总或二次处理。
    """
    result: Dict[str, Any] = {
        "g_file": g_path.name,
        "status": "skip",
        "shapes": 0,
        "message": "",
        "results": [],
    }

    image = cv2.imread(str(image_path))
    if image is None:
        result["message"] = "无法加载图片"
        logger.warning("无法加载图片: %s", image_path)
        return result

    logger.info("=" * 50)
    logger.info("处理 G 文件: %s", g_path.name)
    logger.info("对应图片 : %s", image_path.name)
    logger.info("=" * 50)

    # 根据 G 文件前缀选择算法映射表
    algorithm_map = CIMETYPE_ALGORITHM_MAP_SB if g_path.name.startswith("SB.") else CIMETYPE_ALGORITHM_MAP_SH

    # 解析 G 文件，获取节点树
    all_nodes = FromGstoMapSGZ(
        file_finder,
        str(g_path.parent),
        str(g_path),
        button_config={},
    )
    if not all_nodes:
        result["message"] = "G 文件解析无结果"
        logger.warning("G 文件解析无结果: %s", g_path)
        return result

    ssnodes = all_nodes[0]
    detected = analyze_g_node(image, ssnodes, scheduler, algorithm_map)

    if save_labelme:
        labelme_name = g_path.name.replace(".g", ".json")
        labelme_path = str(output_dir / "labelme" / f"{labelme_name}")
        logger.info("正在生成 LabelMe 标注文件: %s", labelme_path)
        labelme_data = build_labelme_json(
            image_path.name,
            [image.shape[1], image.shape[0]],
            detected,
        )
        with open(labelme_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)
        logger.info("LabelMe 标注已保存: %s (共 %d 个 shapes)", labelme_path, len(labelme_data["shapes"]))
    result["status"] = "success"
    result["shapes"] = len(detected)
    result["results"] = detected
    result["image_path"] = str(image_path)
    result["image_size"] = [image.shape[1], image.shape[0]]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SCADA 画面批量检测脚本：直接解析 G 文件并识别点位"
    )
    parser.add_argument(
        "--image-scada",
        default="/workspace/test/scada/img",
        help="图片输入目录（默认: /workspace/test/scada/img）",
    )
    parser.add_argument(
        "--g-dir",
        default="/workspace/test/scada/g",
        help="G 文件输入目录（默认: /workspace/test/scada/g）",
    )
    parser.add_argument(
        "--tuyuan-dir",
        default="/workspace/test/tuyuan",
        help="图元文件目录（默认: /workspace/test/tuyuan）",
    )
    parser.add_argument(
        "--scada-output",
        default="/workspace/test/scada/result",
        help="LabelMe 结果输出目录（默认: /workspace/test/scada/result）",
    )
    parser.add_argument(
        "--config",
        default="/workspace/scadaandqz/config.json",
        help="模型配置 JSON 路径（默认: /workspace/scadaandqz/config.json）",
    )

    parser.add_argument(
        "--save_labelme",
        action="store_true",
        help="同时输出 LabelMe 格式的 JSON 标注文件（便于在 LabelMe 中可视化查看）",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_scada)
    g_dir = Path(args.g_dir)
    output_dir = Path(args.scada_output)
    tuyuan_dir = Path(args.tuyuan_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    if not g_dir.is_dir():
        raise FileNotFoundError(f"G 文件目录不存在: {g_dir}")
    if not tuyuan_dir.is_dir():
        raise FileNotFoundError(f"图元目录不存在: {tuyuan_dir}")

    # 加载模型配置
    logger.info("加载模型配置: %s", args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    scada_model_config = config.get("scada_model_paths", {})
    if not scada_model_config:
        raise ValueError("配置文件中未找到 scada_model_paths 节点")

    # 初始化算法调度器
    logger.info("正在初始化 YOLO 模型...")
    scheduler = AlgorithmScheduler(scada_model_config)
    logger.info("模型初始化完成，已加载: %s", list(scheduler.algorithm_instances.keys()))

    # 构建图元文件索引（G 目录 + 图元目录）
    file_finder = SimpleFileFinder()
    file_finder.add_dirs([str(g_dir), str(tuyuan_dir)])

    # 遍历所有 图片 文件
    image_extensions = {".jpg",".jpeg",".png",".bmp"}
    img_files = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in image_extensions])
    # g_files = sorted(g_dir.glob("*.g"))
    logger.info("发现 %d 个 图片 文件", len(img_files))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.save_labelme:
        os.makedirs(os.path.join(output_dir, "labelme"), exist_ok=True)

    for img_idx, img_path in enumerate(img_files, start=1):
        g_path = g_dir / (img_path.stem + ".g")
        if not g_path.exists():
            logger.warning("未找到图片 %s 对应的G文件", img_path)
            continue
        try:
            if output_dir is not None:
                result_json_path = str(output_dir / f"{img_path.stem}.json")
            logger.info("=" * 50)
            logger.info("[%d/%d] 开始分析图片: %s", img_idx, len(img_files), img_path.name)
            logger.info("=" * 50)
            t1 = time.time()
            result = process_g_file(g_path, img_path, output_dir, file_finder, scheduler, args.save_labelme)
            t2 = time.time()
            print(f'识别耗时：{t2-t1}')

            if result["status"] == "skip":
                continue

            output_data = {
            "image_path": result["image_path"],
            "image_size": result["image_size"],
            "g_file": result["g_file"],
            "data": result.get("results", {}),
            }
            
            # ---------- 6. 保存结果 ----------
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        except Exception:
            logger.exception("处理 G 文件失败: %s", g_path)


    logger.info("=" * 50)
    logger.info("全部处理完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
