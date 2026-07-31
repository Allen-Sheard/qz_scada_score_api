#!/usr/bin/env python3
"""
前置画面独立检测脚本
====================
输入一张或多张前置画面（表格数据）图片，识别表格中的点号和对应数值。

用法:
    # 单张图片
    python detect_qz.py \\
        --image /path/to/qz.jpg \\
        [--config /workspace/scadaandqz/config.json] \\
        [--idname "数据点号"] \\
        [--valuename "数据值"] \\
        [--ycyx yc] \\
        [--output qz_result.json] \\
        [--visualize] \\
        [--save-labelme]

    # 批量检测目录
    python detect_qz.py \\
        --image /path/to/qz_images/ \\
        --output /path/to/output_dir/ \\
        [--visualize] \\
        [--save-labelme]
"""

import os
import argparse
import json
import logging
import re
from pathlib import Path
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scadaandqz.table import keep_only_number_chars,TableLineToCells,Table,get_point_and_value_coords,crop_same_col_below

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detect_qz")

# ---------- 颜色配置（BGR，用于 OpenCV 画框）----------
COLOR_TABLE_BORDER = (0, 165, 255)      # 橙色 — 表格外框
COLOR_CELL_LINE = (0, 255, 255)         # 黄色 — 单元格线
COLOR_TEXT_BOX = (0, 255, 0)            # 绿色 — 文字检测框
COLOR_ID_CELL = (255, 0, 0)             # 蓝色 — 点号列
COLOR_VALUE_CELL = (0, 0, 255)          # 红色 — 数值列
COLOR_HEADER = (255, 0, 255)            # 紫色 — 表头行

# ---------- 受支持的图片扩展名 ----------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_qz_annotations(
    image: np.ndarray,
    result: Dict[str, Any],
    table_info: Optional[Dict] = None,
    h_ys: Optional[List[int]] = None,
    v_xs: Optional[List[int]] = None,
    cell_boxes_2d: Optional[List[List[List[int]]]] = None,
    id_coord: Optional[Tuple[int, int]] = None,
    value_coord: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    在前置画面图片上绘制可视化标注。
    包括：表格外框、单元格线、表头高亮、点号/数值列高亮、识别结果文字。
    """
    
    pil_image = Image.fromarray(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # 加载字体
    font = None
    font_candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 12)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    # 表格偏移
    offset_x = table_info["bbox"][0] if table_info and "bbox" in table_info else 0
    offset_y = table_info["bbox"][1] if table_info and "bbox" in table_info else 0

    # 1. 绘制表格外框
    if table_info and "bbox" in table_info:
        x1, y1, x2, y2 = table_info["bbox"]
        color = (COLOR_TABLE_BORDER[2], COLOR_TABLE_BORDER[1], COLOR_TABLE_BORDER[0])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"表格 conf={table_info.get('conf', 0):.2f}"
        bbox = draw.textbbox((x1, y1 - 16), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 0, 0))
        draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font)

    # 2. 绘制单元格线
    if h_ys and v_xs and table_info and "bbox" in table_info:
        tx, ty = offset_x, offset_y
        tw = table_info["bbox"][2] - tx
        th = table_info["bbox"][3] - ty
        color = (COLOR_CELL_LINE[2], COLOR_CELL_LINE[1], COLOR_CELL_LINE[0])
        for y in h_ys:
            draw.line([(tx, ty + y), (tw, ty + y)], fill=color, width=1)
        for x in v_xs:
            draw.line([(tx + x, ty), (tx + x, th)], fill=color, width=1)

    # 3. 高亮表头行
    header_row = None
    if id_coord:
        header_row = id_coord[0]
    elif value_coord:
        header_row = value_coord[0]

    if header_row is not None and cell_boxes_2d and 0 <= header_row < len(cell_boxes_2d):
        color = (COLOR_HEADER[2], COLOR_HEADER[1], COLOR_HEADER[0])
        for col_idx, cell in enumerate(cell_boxes_2d[header_row]):
            cx1, cy1, cx2, cy2 = cell[0] + offset_x, cell[1] + offset_y, cell[2] + offset_x, cell[3] + offset_y
            draw.rectangle([cx1, cy1, cx2, cy2], outline=color, width=2)

    # 4. 高亮点号列 & 数值列，并标注 OCR 结果
    qz_yc = result.get("Res", {}).get("qz_yc", [])
    qz_yx = result.get("Res", {}).get("qz_yx", [])
    total_records = qz_yc + qz_yx
    if cell_boxes_2d:
        # 点号列
        if id_coord:
            id_col = id_coord[1]
            color_id = (COLOR_ID_CELL[2], COLOR_ID_CELL[1], COLOR_ID_CELL[0])
            for row_idx, row_cells in enumerate(cell_boxes_2d):
                if 0 <= id_col < len(row_cells):
                    cell = row_cells[id_col]
                    cx1, cy1, cx2, cy2 = cell[0] + offset_x, cell[1] + offset_y, cell[2] + offset_x, cell[3] + offset_y
                    draw.rectangle([cx1, cy1, cx2, cy2], outline=color_id, width=2)
                    # 标注文字
                    try:
                        if row_idx == 0:
                            continue
                        text = total_records[row_idx-1].get('id', '')
                    except Exception:
                        continue
                    if text:
                        tb = draw.textbbox((cx1 + 2, cy1 + 2), text, font=font)
                        draw.rectangle([tb[0]-1, tb[1]-1, tb[2]+1, tb[3]+1], fill=(0, 0, 0))
                        draw.text((cx1 + 2, cy1 + 2), text, fill=(255, 255, 255), font=font)

        # 数值列
        if value_coord:
            val_col = value_coord[1]
            color_val = (COLOR_VALUE_CELL[2], COLOR_VALUE_CELL[1], COLOR_VALUE_CELL[0])
            for row_idx, row_cells in enumerate(cell_boxes_2d):
                if 0 <= val_col < len(row_cells):
                    cell = row_cells[val_col]
                    cx1, cy1, cx2, cy2 = cell[0] + offset_x, cell[1] + offset_y, cell[2] + offset_x, cell[3] + offset_y
                    draw.rectangle([cx1, cy1, cx2, cy2], outline=color_val, width=2)
                    # 标注文字
                    try:
                        if row_idx == 0:
                            continue
                        text = total_records[row_idx-1].get('value', '')
                    except Exception:
                        text = "."
                    if text:
                        tb = draw.textbbox((cx1 + 2, cy1 + 2), text, font=font)
                        draw.rectangle([tb[0]-1, tb[1]-1, tb[2]+1, tb[3]+1], fill=(0, 0, 0))
                        draw.text((cx1 + 2, cy1 + 2), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _match_records_with_row(
    id_labels: Dict[Tuple[int, int], str],
    value_labels: Dict[Tuple[int, int], str],
    id_coord: Tuple[int, int],
    value_coord: Tuple[int, int],
) -> List[Tuple[int, str, str]]:
    """
    按行匹配 ID 和 Value，并保留原始行号。
    返回: [(row, id_value, value_value), ...]
    """
    row_numbers = set(k[0] for k in id_labels.keys())
    _, id_col = id_coord
    _, value_col = value_coord
    records: List[Tuple[int, str, str]] = []
    for row in sorted(row_numbers):
        id_value = id_labels.get((row, id_col), "").strip()
        # 保留 ID 开头的负号，仅删除其他非数字字符（避免 "-1" 被清洗成 "1"）
        is_negative = id_value.startswith("-")
        id_value = re.sub(r"[^0-9]", "", id_value)
        if is_negative and id_value:
            id_value = "-" + id_value
        value_value = value_labels.get((row, value_col), "").strip()
        if value_value and value_value.lower() in {"o", "c", "n"}:
            value_value = "0"
        value_value = keep_only_number_chars(value_value)
        if value_value == "":
            value_value = "0"
        if id_value and value_value:
            records.append((row, id_value, value_value))
    return records


def save_labelme_json(
    output_path: str,
    image_path: str,
    image_size: List[int],
    type: str,
    records_with_row: List[Tuple[int, str, str]],
    cell_boxes_2d: Optional[List[List[List[int]]]],
    value_coord: Optional[Tuple[int, int]],
    table_info: Optional[Dict] = None,
) -> None:
    """
    保存为 LabelMe 格式。
    每个 shape 的框坐标为 valuename 单元格位置，label 为 id_value。
    """
    if value_coord is None or not cell_boxes_2d:
        logger.warning("无法生成 LabelMe 标注：缺少数值列坐标或单元格信息")
        return

    offset_x = table_info["bbox"][0] if table_info and "bbox" in table_info else 0
    offset_y = table_info["bbox"][1] if table_info and "bbox" in table_info else 0
    value_col = value_coord[1]

    shapes = []
    for row, rid, rval in records_with_row:
        if row < 0 or row >= len(cell_boxes_2d):
            continue
        if value_col < 0 or value_col >= len(cell_boxes_2d[row]):
            continue
        cell = cell_boxes_2d[row][value_col]
        x1, y1, x2, y2 = (
            cell[0] + offset_x,
            cell[1] + offset_y,
            cell[2] + offset_x,
            cell[3] + offset_y,
        )
        shapes.append(
            {
                "label": f"{rid}_{rval}_{type}",
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(Path(image_path).name),
        "imageData": None,
        "imageHeight": image_size[1],
        "imageWidth": image_size[0],
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)
    logger.info("LabelMe 标注已保存: %s", output_path)


def analyze_qz_image(
    image_path: str,
    ocr_recognizer,
    params: Dict[str, str]
) -> Dict[str, Any]:
    """
    分析前置画面主入口。
    为了保留中间可视化信息，手动执行 OCRRecognizer.run() 的内部步骤。
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"无法加载图片: {image_path}")
    logger.info("图片加载成功: %s (%dx%d)", image_path, frame.shape[1], frame.shape[0])

    vis_info = {
        "table_info": None,
        "h_ys": None,
        "v_xs": None,
        "cell_boxes_2d": None,
        "id_coord": None,
        "value_coord": None,
        "id_labels": {},
        "value_labels": {},
    }

    # 参数解析
    ValueNameyc = params.get("valuenameyc", "")
    ValueNameyx = params.get("valuenameyx", "")
    ycyx = params.get("ycyx", "yc")
    ValueName = ValueNameyc if ycyx == "yc" else ValueNameyx
    IDname = params.get("idname", "数据点号")

    # Step 1: 表格检测
    logger.info("正在检测表格...")
    h_ys, v_xs, table_img = ocr_recognizer.tabelmodel.run(frame)
    # try:
    #     cv2.imwrite(table_debug_path, table_img)
    # except Exception as e:
    #     logger.warning("保存表格调试图失败 %s: %s", table_debug_path, e)
    if h_ys is None or v_xs is None:
        logger.warning("未检测到表格")
        return {
            "image_path": image_path,
            "image_size": [frame.shape[1], frame.shape[0]],
            "params": params,
            "result": {"Res": {"qz_yc": [], "qz_yx": []}, "status": "fail", "error": "未检测到表格"},
            "records_with_row": [],
            "visualization": vis_info,
        }

    # 保存表格外框
    best_table = ocr_recognizer.tabelmodel.detect_best_table(frame)
    vis_info["table_info"] = best_table
    vis_info["h_ys"] = h_ys
    vis_info["v_xs"] = v_xs

    # Step 2: 文字检测 & 区域检测
    logger.info("正在检测文字框...")
    text_polys = ocr_recognizer.TextDetModel.predict(table_img)[0]["dt_polys"]
    text_boxs = ocr_recognizer.wzqymodel.predict(table_img)

    # Step 3: 生成单元格
    tabelcline = TableLineToCells(h_ys, v_xs, table_img.shape[:2])
    cell_boxes_2d = tabelcline.generate_cell_boxes_2d()
    vis_info["cell_boxes_2d"] = cell_boxes_2d
    logger.info("表格单元格: %d 行 x %d 列", len(cell_boxes_2d),
                len(cell_boxes_2d[0]) if cell_boxes_2d else 0)

    # Step 4: 构建 Table 对象 & 识别表头
    ta = Table(table_img, cell_boxes_2d, text_polys, text_boxs)
    ID_coord, value_coord = get_point_and_value_coords(
        ta, cell_boxes_2d, ocr_recognizer.TextRecModel,
        max_check_rows=3, IDname=IDname, ValueName=ValueName
    )
    vis_info["id_coord"] = ID_coord
    vis_info["value_coord"] = value_coord
    logger.info("表头坐标: ID=%s, Value=%s", ID_coord, value_coord)

    if ID_coord is None or value_coord is None:
        return {
            "image_path": image_path,
            "image_size": [frame.shape[1], frame.shape[0]],
            "params": params,
            "result": {"Res": {"qz_yc": [], "qz_yx": []}, "status": "fail",
                       "error": f"未找到标头 {IDname} 或 {ValueName}"},
            "records_with_row": [],
            "visualization": vis_info,
        }

    # Step 5: 识别点号列 & 数值列
    logger.info("正在识别点号列...")
    ID_labels = crop_same_col_below(ta, cell_boxes_2d, ID_coord, ocr_recognizer.TextRecModel, qz=True)
    vis_info["id_labels"] = ID_labels
    # print(f'ID:{ID_labels}')
    logger.info("正在识别数值列...")
    value_labels = crop_same_col_below(ta, cell_boxes_2d, value_coord, ocr_recognizer.TextRecModel, qz=True)
    vis_info["value_labels"] = value_labels
    # print(f'value:{value_labels}')

    # Step 6: 匹配结果
    records_with_row = _match_records_with_row(ID_labels, value_labels, ID_coord, value_coord)
    reslist = [{"id": rid, "value": rval} for _, rid, rval in records_with_row]

    res = {"qz_yc": [], "qz_yx": []}
    if ycyx == "yc":
        res["qz_yc"] = reslist
    else:
        res["qz_yx"] = reslist

    result = {"Res": res, "status": "success", "error": ""}
    logger.info("识别完成，共 %d 条记录", len(reslist))

    return {
        "image_path": image_path,
        "image_size": [frame.shape[1], frame.shape[0]],
        "params": params,
        "result": result,
        "records_with_row": records_with_row,
        "visualization": vis_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="前置画面独立检测脚本：识别表格中的点号和数值"
    )
    parser.add_argument("--image-qz", default="test/qz/img/", help="输入前置画面图片路径或图片目录")
    parser.add_argument(
        "--config",
        default="/workspace/scadaandqz/config.json",
        help="模型配置 JSON 路径（默认: /workspace/scadaandqz/config.json）",
    )
    parser.add_argument(
        "--idname",
        default="数据点号",
        help="表头中点号列的名称（默认: 数据点号）",
    )
    parser.add_argument(
        "--valuename",
        default="数据值",
        help="表头中数值列的名称（默认: 数据值）",
    )
    parser.add_argument(
        "--ycyx",
        default="yc",
        choices=["yc", "yx"],
        help="数据类型: yc=遥测 / yx=遥信（默认: yc）",
    )
    parser.add_argument(
        "--qz-output",
        default="test/qz/result/",
        help="输出结果 JSON 路径或输出目录（批量检测时必须是目录）",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="生成可视化标注图片",
    )
    parser.add_argument(
        "--save-labelme",
        action="store_true",
        help="保存 LabelMe 格式标注文件（框坐标为 valuename 位置，name 为 id_value）",
    )

    args = parser.parse_args()

    # ---------- 1. 加载配置 ----------
    logger.info("加载模型配置: %s", args.config)
    config = load_config(args.config)
    table_model_config = config.get("table_model_paths", {})
    if not table_model_config:
        raise ValueError("配置文件中未找到 table_model_paths 节点")

    # ---------- 2. 初始化 OCR 识别器 ----------
    logger.info("正在初始化 OCR 模型（PP-OCRv5 + YOLO11）...")
    # 动态导入，避免在脚本顶部就加载重模型
    from scadaandqz.table import OCRRecognizer
    ocr = OCRRecognizer(table_model_config)
    logger.info("OCR 模型初始化完成")

    # ---------- 3. 构建识别参数 ----------
    params = {
        "idname": args.idname,
        "valuenameyc": args.valuename if args.ycyx == "yc" else "",
        "valuenameyx": args.valuename if args.ycyx == "yx" else "",
        "ycyx": args.ycyx,
    }
    logger.info("识别参数: idname=%s, valuename=%s, type=%s", args.idname, args.valuename, args.ycyx)

    # ---------- 4. 确定输入图片列表 ----------
    input_path = Path(args.image_qz)
    if input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"目录中未找到图片: {args.image_qz}")
        is_batch = True
        logger.info("批量输入目录: %s，共发现 %d 张图片", args.image_qz, len(image_paths))
    else:
        image_paths = [input_path]
        is_batch = False

    # ---------- 5. 确定输出目录/文件 ----------
    output_arg = Path(args.qz_output)
    if is_batch:
        if args.qz_output == "qz_detect_result.json":
            output_dir = input_path.parent / (input_path.name + "_results")
        else:
            output_dir = output_arg
            if output_dir.suffix:
                raise ValueError("批量检测时 --output 必须是目录")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_for_single = output_dir
        result_json_path_for_single = None
    else:
        if output_arg.suffix == "":
            output_dir_for_single = output_arg
            output_dir_for_single.mkdir(parents=True, exist_ok=True)
            result_json_path_for_single = None
        else:
            output_dir_for_single = None
            result_json_path_for_single = args.output

    # ---------- 6. 批量/单张处理 ----------
    total_images = len(image_paths)
    total_records_all = 0
    for img_idx, img_path in enumerate(image_paths, start=1):
        stem = img_path.stem

        if output_dir_for_single is not None:
            result_json_path = str(output_dir_for_single / f"{stem}.json")
            vis_img_path = str(output_dir_for_single / f"{stem}.jpg") if args.visualize else None
            labelme_json_path = str(output_dir_for_single / "labelme" / f"{stem}.json") if args.save_labelme else None
        else:
            result_json_path = result_json_path_for_single
            base = str(Path(result_json_path).with_suffix(""))
            vis_img_path = base + "_vis.jpg" if args.visualize else None
            labelme_json_path = base + "_labelme.json" if args.save_labelme else None
        print("[%d/%d] 开始分析图片: %s", img_idx, total_images, img_path)
        logger.info("=" * 50)
        logger.info("[%d/%d] 开始分析图片: %s", img_idx, total_images, img_path)
        logger.info("=" * 50)

        t1 = time.time()
        analysis = analyze_qz_image(
            str(img_path), ocr, params
        )
        t2 = time.time()
        print(f'识别耗时：{t2-t1}')
        result = analysis["result"]

        # 保存结果 JSON
        output_data = {
            "image_path": analysis["image_path"],
            "image_size": analysis["image_size"],
            "params": analysis["params"],
            "status": result.get("status", "fail"),
            "error": result.get("error", ""),
            "data": result.get("Res", {}),
        }
        out_dir = os.path.dirname(result_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 保存可视化标注
        if vis_img_path:
            logger.info("正在生成可视化标注图: %s", vis_img_path)
            vis_image = cv2.imread(str(img_path))
            if vis_image is not None:
                vis = draw_qz_annotations(
                    vis_image,
                    result,
                    table_info=analysis["visualization"].get("table_info"),
                    h_ys=analysis["visualization"].get("h_ys"),
                    v_xs=analysis["visualization"].get("v_xs"),
                    cell_boxes_2d=analysis["visualization"].get("cell_boxes_2d"),
                    id_coord=analysis["visualization"].get("id_coord"),
                    value_coord=analysis["visualization"].get("value_coord")
                )
                cv2.imwrite(vis_img_path, vis)
                logger.info("可视化图片已保存: %s", vis_img_path)
            else:
                logger.warning("无法重新加载图片用于可视化")

        # 保存 LabelMe 格式
        if labelme_json_path:
            save_labelme_json(
                labelme_json_path,
                str(img_path),
                analysis["image_size"],
                analysis["params"].get("ycyx"),
                analysis.get("records_with_row", []),
                analysis["visualization"].get("cell_boxes_2d"),
                analysis["visualization"].get("value_coord"),
                analysis["visualization"].get("table_info"),
            )

        # 终端摘要
        logger.info("-" * 50)
        logger.info("[%d/%d] 分析完成！", img_idx, total_images)
        logger.info("状态: %s", result.get("status", "unknown"))
        if result.get("error"):
            logger.info("错误: %s", result["error"])

        qz_yc = result.get("Res", {}).get("qz_yc", [])
        qz_yx = result.get("Res", {}).get("qz_yx", [])
        total_records = len(qz_yc) + len(qz_yx) # type: ignore
        total_records_all += total_records
        logger.info("识别记录数: %d (yc=%d, yx=%d)", total_records, len(qz_yc), len(qz_yx))
        logger.info("结果已保存: %s", result_json_path)
        if vis_img_path:
            logger.info("可视化图片: %s", vis_img_path)
        if labelme_json_path:
            logger.info("LabelMe 标注: %s", labelme_json_path)
        logger.info("-" * 50)

        # 打印预览
        # if qz_yc or qz_yx:
        #     print("\n【识别结果】")
        #     for item in (qz_yc + qz_yx):
        #         print(f"  ID: {item.get('id', ''):20s} | Value: {item.get('value', '')}")

    logger.info("=" * 50)
    logger.info("全部处理完成！共 %d 张图片，总识别记录数 %d", total_images, total_records_all)
    if output_dir_for_single is not None:
        logger.info("输出目录: %s", output_dir_for_single)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
