# -*- coding: utf-8 -*-
"""
打分与错误可视化接口，供外部程序调用。

提供：
    - ScoreEngine: 对比 QZ / SCADA 参考答案与预测结果，
      生成准确率报告，并在对应图片上绘制错误框。
"""

import sys
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 确保项目根目录在路径中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ===================== 颜色常量（BGR，OpenCV 使用）=====================
COLOR_RED = (0, 0, 255)          # 不匹配
COLOR_PURPLE = (255, 0, 255)     # 缺失（只有答案，无预测）
COLOR_YELLOW = (0, 255, 255)     # 多余
COLOR_ORANGE = (0, 140, 255)     # 识别失败
COLOR_GREEN = (0, 255, 0)        # 正确（可选）
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


# ===================== 通用工具函数 =====================
def is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def rel_error_ok(ref: str, pred: str, threshold: float = 0.01) -> bool:
    """判断相对误差是否小于阈值；无法转数值时退化为精确匹配。"""
    if not is_numeric(ref) or not is_numeric(pred):
        return str(ref) == str(pred)
    ref_f = float(ref)
    pred_f = float(pred)
    if ref_f == 0:
        return pred_f == 0
    return abs((pred_f - ref_f) / ref_f) < threshold


def points_to_box(points: List[List[float]]) -> Optional[List[int]]:
    """
    LabelMe 的 points 可能是 [[x1,y1],[x2,y2]] 矩形，
    也可能是多边形。返回 [x1,y1,x2,y2] 整数框。
    """
    if not points or len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_font(size: int = 12):
    """加载中文字体。"""
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ===================== QZ 解析与对比 =====================
def parse_labelme_qz(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """解析 QZ LabelMe 参考答案，返回记录列表（含 box）与 shape 数量。"""
    data = load_json(path)
    shapes = data.get("shapes", [])
    records = []
    for shape in shapes:
        label = shape.get("label", "")
        if not label:
            continue
        parts = label.split("_")
        if len(parts) != 3:
            continue
        rid, rval, rtype = parts
        box = points_to_box(shape.get("points", []))
        records.append({
            "id": rid.strip(),
            "value": rval.strip(),
            "type": rtype.strip(),
            "box": box,
        })
    return records, len(shapes)


def parse_pred_qz(path: Path) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """解析 QZ 预测结果 JSON，返回类型与记录列表（含 box）。"""
    data = load_json(path)
    data_field = data.get("data", {})
    if data_field.get("qz_yc"):
        return "qz_yc", data_field["qz_yc"]
    elif data_field.get("qz_yx"):
        return "qz_yx", data_field["qz_yx"]
    return None, []


def compare_record_qz(ref: Dict[str, Any], pred: Dict[str, Any], mode: str) -> bool:
    """QZ 单条记录是否正确。"""
    if mode == "qz_yc":
        return ref["id"] == pred["id"] and ref["value"] == pred["value"]
    else:  # qz_yx
        if ref["value"] == pred["value"]:
            return ref["id"] == pred["id"]
        return ref["id"] == pred["id"] and rel_error_ok(ref["value"], pred["value"])


def compare_file_qz(
    ref_records: List[Dict[str, Any]],
    pred_type: str,
    pred_records: List[Dict[str, Any]],
    threshold: float = 0.01,
) -> Tuple[int, List[Dict[str, Any]]]:
    """QZ 单文件对比，返回正确数与带 box 的错误列表。"""
    pred_by_id = {str(r.get("id", "")): r for r in pred_records}
    correct = 0
    errors = []

    for ref in ref_records:
        ref_id = ref["id"]
        pred = pred_by_id.get(ref_id)
        if not pred:
            errors.append({
                "type": "missing",
                "ref_id": ref_id,
                "ref_value": ref["value"],
                # "ref_ycyx": ref.get("type", ""),
                "ref_box": ref.get("box"),
                "pred_id": None,
                "pred_value": None,
                "pred_box": None,
                "message": f"缺失：ID={ref_id}, 期望={ref['value']}",
            })
            continue

        if compare_record_qz(ref, pred, pred_type):
            correct += 1
        else:
            errors.append({
                "type": "mismatch",
                "ref_id": ref_id,
                "ref_value": ref["value"],
                # "ref_ycyx": ref.get("type", ""),
                "ref_box": ref.get("box"),
                "pred_id": pred.get("id"),
                "pred_value": pred.get("value"),
                "pred_box": pred.get("box"),
                "message": f"不匹配：ID={ref_id}, 期望={ref['value']}, 识别={pred.get('value')}",
            })

    return correct, errors


# ===================== SCADA 解析与对比 =====================
def parse_labelme_scada(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """解析 SCADA LabelMe 参考答案，返回记录列表（含 box）与 shape 数量。"""
    data = load_json(path)
    shapes = data.get("shapes", [])
    records = []
    for shape in shapes:
        label = shape.get("label", "")
        if not label:
            continue
        if label.endswith("_yc"):
            ycyx = "yc"
        elif label.endswith("_yx"):
            ycyx = "yx"
        else:
            continue
        # label 格式 id_value_ycyx，第一个下划线分隔 id 与 value
        parts = label.split("_")
        if len(parts) < 3:
            continue
        rid = parts[0]
        rval = parts[1]
        box = points_to_box(shape.get("points", []))
        records.append({
            "id": rid,
            "value": rval,
            "ycyx": ycyx,
            "box": box,
        })
    return records, len(shapes)


def parse_pred_scada(path: Path) -> List[Dict[str, Any]]:
    """解析 SCADA 预测结果 JSON，返回 data 数组。"""
    data = load_json(path)
    return data.get("data", []) or []


def compare_record_scada(ref: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    """SCADA 单条记录是否正确。"""
    pred_id = str(pred.get("cimeid", "")).strip()
    pred_value = str(pred.get("value", "")).strip()
    pred_ycyx = str(pred.get("ycyx", "")).strip().lower()

    if ref["id"] != pred_id:
        return False

    if pred_ycyx == "yx":
        return ref["value"] == pred_value
    else:  # yc
        return rel_error_ok(ref["value"], pred_value)


def compare_file_scada(
    ref_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """SCADA 单文件对比，返回正确数与带 box 的错误列表。"""
    pred_by_id = {str(r.get("cimeid", "")).strip(): r for r in pred_records}
    correct = 0
    errors = []

    for ref in ref_records:
        ref_id = ref["id"]
        pred = pred_by_id.get(ref_id)
        if not pred:
            errors.append({
                "type": "missing",
                "ref_id": ref_id,
                "ref_value": ref["value"],
                # "ref_ycyx": ref["ycyx"],
                "ref_box": ref.get("box"),
                "pred_id": None,
                "pred_value": None,
                "pred_box": None,
                "message": f"缺失：ID={ref_id}, 期望={ref['value']}",
            })
            continue

        if compare_record_scada(ref, pred):
            correct += 1
        else:
            errors.append({
                "type": "mismatch",
                "ref_id": ref_id,
                "ref_value": ref["value"],
                # "ref_ycyx": ref["ycyx"],
                "ref_box": ref.get("box"),
                "pred_id": pred.get("cimeid"),
                "pred_value": pred.get("value"),
                # "pred_ycyx": pred.get("ycyx"),
                "pred_box": pred.get("roi"),
                "message": f"不匹配：ID={ref_id}, 期望={ref['value']}, 识别={pred.get('value')}",
            })

    # 可选：统计预测中多余的点（当前未作为错误项，如需可启用）
    # ref_ids = {r["id"] for r in ref_records}
    # for pred in pred_records:
    #     pred_id = str(pred.get("cimeid", "")).strip()
    #     if pred_id and pred_id not in ref_ids:
    #         errors.append({
    #             "type": "extra",
    #             "ref_id": None,
    #             "ref_value": None,
    #             "ref_ycyx": None,
    #             "ref_box": None,
    #             "pred_id": pred_id,
    #             "pred_value": pred.get("value"),
    #             "pred_ycyx": pred.get("ycyx"),
    #             "pred_box": pred.get("roi"),
    #             "message": f"多余：ID={pred_id}, 识别={pred.get('value')}",
    #         })

    return correct, errors


# ===================== 画框工具 =====================
def _draw_label_with_bg(
    draw: ImageDraw.Draw,
    text: str,
    xy: Tuple[int, int],
    font: ImageFont.FreeTypeFont,
    text_color: Tuple[int, int, int] = COLOR_WHITE,
    bg_color: Tuple[int, int, int] = COLOR_BLACK,
) -> None:
    """在指定位置绘制带背景的文字标签。"""
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=bg_color)
    draw.text(xy, text, fill=text_color, font=font)


def draw_boxes_on_image(
    image_path: str,
    boxes: List[Dict[str, Any]],
    save_path: str,
) -> str:
    """
    在图片上绘制多个带标签的矩形框。

    参数:
        image_path: 原图路径
        boxes: 每个元素 dict，包含：
            - box: [x1, y1, x2, y2]
            - color: (B, G, R) 元组
            - label: str
        save_path: 保存路径

    返回:
        保存路径
    """
    image_path = Path(image_path).resolve()
    save_path = Path(save_path).resolve()

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = _get_font(12)

    # 按框的 y 坐标排序，避免标签堆叠
    sorted_boxes = sorted(boxes, key=lambda b: (b["box"][1], b["box"][0]))

    last_y = -999
    for item in sorted_boxes:
        box = item["box"]
        color = item.get("color", COLOR_RED)
        label = item.get("label", "")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        # PIL 颜色为 RGB
        rgb_color = (color[2], color[1], color[0])
        draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=2)

        if label:
            # 标签位置：框上方，若太近则向下错开
            text_y = y1 - 14
            if abs(text_y - last_y) < 14 and text_y > 0:
                text_y = y1 + 2
            last_y = text_y
            _draw_label_with_bg(draw, label, (x1, max(0, text_y)), font, text_color=COLOR_WHITE, bg_color=rgb_color)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
    return str(save_path)

# ===================== ScoreEngine 主类 =====================
class ScoreEngine:
    """
    打分与错误可视化引擎。

    用法：
        engine = ScoreEngine()
        report = engine.score_qz(
            ref_dir="/workspace/test/daan/qz",
            pred_dir="/workspace/test/qz/result",
            output_dir="/workspace/test/qz/score",
            image_dir="/workspace/test/qz/img",
        )
    """

    def __init__(self, rel_error_threshold: float = 0.01):
        self.rel_error_threshold = rel_error_threshold

    # ---------- QZ 打分 ----------
    def score_qz(
        self,
        ref_dir: str,
        pred_dir: str,
        output_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        pred_ext: str = ".json",
    ) -> Dict[str, Any]:
        """
        对 QZ 结果进行打分并生成错误可视化图。

        参数:
            ref_dir: LabelMe 参考答案 JSON 目录
            pred_dir: QZAnalyzer 或 detect_qz 生成的预测结果 JSON 目录
            output_dir: 可选，报告与错误图输出目录
            image_dir: 可选，原图目录，用于绘制错误框
            pred_ext: 预测文件后缀，默认 .json

        返回:
            报告字典，包含 summary、file_reports、error_files
        """
        ref_dir = Path(ref_dir).resolve()
        pred_dir = Path(pred_dir).resolve()
        if output_dir:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        if image_dir:
            image_dir = Path(image_dir).resolve()

        if not ref_dir.is_dir():
            raise FileNotFoundError(f"参考答案目录不存在: {ref_dir}")
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"预测结果目录不存在: {pred_dir}")

        ref_files = sorted(ref_dir.glob("*.json"))
        total_ref_points = 0
        total_correct = 0
        file_reports = []
        error_files = []

        for ref_path in ref_files:
            pred_path = pred_dir / (ref_path.stem + pred_ext)
            file_report = {
                "file": ref_path.name,
                "ref_path": str(ref_path),
                "pred_path": str(pred_path) if pred_path.exists() else None,
            }

            if not pred_path.exists():
                file_report["status"] = "pred_missing"
                file_report["errors"] = []
                error_files.append(file_report)
                continue

            ref_records, ref_count = parse_labelme_qz(ref_path)
            pred_type, pred_records = parse_pred_qz(pred_path)
            if pred_type is None:
                file_report["status"] = "no_data"
                file_report["errors"] = []
                error_files.append(file_report)
                continue

            pred_count = len(pred_records)
            total_ref_points += ref_count
            correct, errors = compare_file_qz(ref_records, pred_type, pred_records, self.rel_error_threshold)
            total_correct += correct

            file_report.update({
                "status": "ok" if not errors else "error",
                "ref_count": ref_count,
                "pred_count": pred_count,
                "correct": correct,
                "errors": errors,
            })

            if errors:
                error_files.append(file_report)
                if image_dir and output_dir:
                    # 去掉 .json 后缀得到基础名（支持 .pic.json 等多层后缀）
                    base_name = ref_path.with_suffix("").name
                    img_path = image_dir / (base_name + ".jpg")
                    if not img_path.exists():
                        # 尝试其他后缀
                        for ext in (".jpeg", ".png", ".bmp"):
                            img_path = image_dir / (base_name + ext)
                            if img_path.exists():
                                break
                    if img_path.exists():
                        save_path = output_dir / f"{base_name}_error.jpg"
                        try:
                            self.draw_errors_on_image(str(img_path), errors, str(save_path), mode="qz")
                            file_report["error_image"] = str(save_path)
                        except Exception as e:
                            logger.warning("绘制错误图失败 %s: %s", img_path, e)

            file_reports.append(file_report)

        accuracy = total_correct / total_ref_points if total_ref_points > 0 else 0.0
        summary = {
            "mode": "qz",
            "total_ref_points": total_ref_points,
            "total_correct": total_correct,
            "accuracy": round(accuracy, 4),
            "accuracy_percent": f"{round(accuracy * 100, 2)}%",
            "total_files": len(ref_files),
            "error_files_count": len(error_files)
        }

        report = {
            "summary": summary,
            "file_reports": file_reports,
            "error_files": error_files,
        }

        if output_dir:
            report_path = output_dir / "report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info("QZ 打分报告已保存: %s", report_path)

        return report

    # ---------- SCADA 打分 ----------
    def score_scada(
        self,
        ref_dir: str,
        pred_dir: str,
        output_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对 SCADA 结果进行打分并生成错误可视化图。

        参数与返回同 score_qz。
        """
        ref_dir = Path(ref_dir).resolve()
        pred_dir = Path(pred_dir).resolve()
        if output_dir:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        if image_dir:
            image_dir = Path(image_dir).resolve()

        if not ref_dir.is_dir():
            raise FileNotFoundError(f"参考答案目录不存在: {ref_dir}")
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"预测结果目录不存在: {pred_dir}")

        ref_files = sorted(ref_dir.glob("*.json"))
        total_ref_points = 0
        total_correct = 0
        file_reports = []
        error_files = []

        for ref_path in ref_files:
            pred_path = pred_dir / ref_path.name
            file_report = {
                "file": ref_path.name,
                "ref_path": str(ref_path),
                "pred_path": str(pred_path) if pred_path.exists() else None,
            }

            if not pred_path.exists():
                file_report["status"] = "pred_missing"
                file_report["errors"] = []
                error_files.append(file_report)
                continue

            ref_records, ref_count = parse_labelme_scada(ref_path)
            pred_records = parse_pred_scada(pred_path)
            pred_count = len(pred_records)
            total_ref_points += ref_count
            correct, errors = compare_file_scada(ref_records, pred_records)
            total_correct += correct

            file_report.update({
                "status": "ok" if not errors else "error",
                "ref_count": ref_count,
                "pred_count": pred_count,
                "correct": correct,
                "errors": errors,
            })

            if errors:
                error_files.append(file_report)
                if image_dir and output_dir:
                    # 去掉 .json 后缀得到基础名（支持 .pic.json 等多层后缀）
                    base_name = ref_path.with_suffix("").name
                    img_path = image_dir / (base_name + ".jpg")
                    # SCADA 图片后缀不定，尝试常见后缀
                    for ext in (".jpeg", ".png", ".bmp"):
                        candidate = image_dir / (base_name + ext)
                        if candidate.exists():
                            img_path = candidate
                            break
                    if img_path.exists():
                        save_path = output_dir / f"{base_name}_error.jpg"
                        try:
                            self.draw_errors_on_image(str(img_path), errors, str(save_path), mode="scada")
                            file_report["error_image"] = str(save_path)
                        except Exception as e:
                            logger.warning("绘制错误图失败 %s: %s", img_path, e)

            file_reports.append(file_report)

        accuracy = total_correct / total_ref_points if total_ref_points > 0 else 0.0
        summary = {
            "mode": "scada",
            "total_ref_points": total_ref_points,
            "total_correct": total_correct,
            "accuracy": round(accuracy, 4),
            "accuracy_percent": f"{round(accuracy * 100, 2)}%",
            "total_files": len(ref_files),
            "error_files_count": len(error_files)
        }

        report = {
            "summary": summary,
            "file_reports": file_reports,
            "error_files": error_files,
        }

        if output_dir:
            report_path = output_dir / "report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info("SCADA 打分报告已保存: %s", report_path)

        return report

    # ---------- 错误框绘制 ----------
    def draw_errors_on_image(
        self,
        image_path: str,
        errors: List[Dict[str, Any]],
        save_path: str,
        mode: str = "qz",
    ) -> str:
        """
        在图片上绘制错误框。

        参数:
            image_path: 原图路径
            errors: 由 score_* 返回的错误项列表
            save_path: 输出图片路径
            mode: "qz" 或 "scada"

        返回:
            保存路径
        """
        boxes = []
        for err in errors:
            etype = err.get("type")
            if etype == "mismatch":
                # 同时画参考答案框与预测框
                ref_box = err.get("ref_box")
                pred_box = err.get("pred_box") or err.get("box")
                if ref_box and len(ref_box) == 4:
                    boxes.append({
                        "box": ref_box,
                        "color": COLOR_PURPLE,
                        "label": f"daan {err.get('ref_id')}={err.get('ref_value')} \n pred {err.get('pred_id')}={err.get('pred_value')}",
                    })
            elif etype == "missing":
                ref_box = err.get("ref_box")
                if ref_box and len(ref_box) == 4:
                    boxes.append({
                        "box": ref_box,
                        "color": COLOR_PURPLE,
                        "label": f"miss {err.get('ref_id')}={err.get('ref_value')}",
                    })
            elif etype == "extra":
                pred_box = err.get("pred_box") or err.get("box")
                if pred_box and len(pred_box) == 4:
                    boxes.append({
                        "box": pred_box,
                        "color": COLOR_YELLOW,
                        "label": f"extra {err.get('pred_id')}={err.get('pred_value')}",
                    })
            elif etype == "fail":
                box = err.get("box") or err.get("pred_box") or err.get("ref_box")
                if box and len(box) == 4:
                    boxes.append({
                        "box": box,
                        "color": COLOR_ORANGE,
                        "label": f"fail {err.get('ref_id', err.get('pred_id'))}",
                    })

        return draw_boxes_on_image(image_path, boxes, save_path)

    # ---------- 报告打印 ----------
    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """打印报告到控制台。"""
        summary = report.get("summary", {})
        error_files = report.get("error_files", [])
        print("=" * 60)
        print("统计结果")
        print("=" * 60)
        print(f"模式                     : {summary.get('mode', '')}")
        print(f"样本总点数（points）      : {summary.get('total_ref_points', 0)}")
        print(f"正确数                   : {summary.get('total_correct', 0)}")
        print(f"正确率                   : {summary.get('accuracy', 0.0):.4f}（{summary.get('accuracy_percent', 0.0):.2f}%）")
        print(f"文件总数                 : {summary.get('total_files', 0)}")
        print(f"错误文件数               : {summary.get('error_files_count', 0)}")
        print("=" * 60)
        if error_files:
            print("\n错误文件：")
            for item in error_files:
                print(f"  文件: {item['file']}  错误数: {len(item.get('errors', []))}")
                if item.get("error_image"):
                    print(f"    错误图: {item['error_image']}")
