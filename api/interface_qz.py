# -*- coding: utf-8 -*-
"""
QZ 前置画面检测接口，供外部程序调用。

提供：
    - QZAnalyzer: 模型一次性加载，单张/批量分析，返回结构化结果。
    - 返回结果中每条记录包含 value 列单元格 box，便于后续错误框可视化。
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

# 确保项目根目录在路径中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from detect_qz import analyze_qz_image, draw_qz_annotations, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = "/workspace/scadaandqz/config.json"

DEFAULT_PARAMS = {
    "idname": "数据点号",
    "valuenameyc": "数据值",
    "valuenameyx": "数据值",
    "ycyx": "yc",
}


class QZAnalyzer:
    """
    QZ 前置画面分析器。

    初始化时加载 OCR 模型，后续可多次调用 analyze 进行分析，避免重复加载。
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        参数:
            config_path: 模型配置文件路径，默认 /workspace/scadaandqz/config.json
        """
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.table_model_config = self.config.get("table_model_paths", {})
        if not self.table_model_config:
            raise ValueError(f"配置文件 {self.config_path} 中未找到 table_model_paths 节点")

        logger.info("QZAnalyzer 初始化：加载 OCR 模型...")
        from scadaandqz.table import OCRRecognizer
        self.ocr = OCRRecognizer(self.table_model_config)
        logger.info("QZAnalyzer 初始化完成")

    def _merge_params(self, params: Optional[Dict[str, str]]) -> Dict[str, str]:
        """合并用户参数与默认参数。"""
        merged = dict(DEFAULT_PARAMS)
        if params:
            merged.update(params)
        return merged

    def _fill_record_boxes(
        self,
        result: Dict[str, Any],
        records_with_row: List[Any],
        visualization: Dict[str, Any],
    ) -> None:
        """
        根据单元格信息为结果中每条记录补充 value 列单元格的 box。
        修改 result["Res"]["qz_yc" / "qz_yx"] 中记录，添加 box 字段。
        """
        cell_boxes_2d = visualization.get("cell_boxes_2d")
        value_coord = visualization.get("value_coord")
        table_info = visualization.get("table_info")
        if not cell_boxes_2d or value_coord is None:
            return

        offset_x = table_info["bbox"][0] if table_info and "bbox" in table_info else 0
        offset_y = table_info["bbox"][1] if table_info and "bbox" in table_info else 0
        val_col = value_coord[1]

        ycyx = result.get("params", {}).get("ycyx", "yc")
        target_list = result.get("Res", {}).get("qz_yc" if ycyx == "yc" else "qz_yx", [])

        for idx, (row, _rid, _rval) in enumerate(records_with_row):
            if idx >= len(target_list):
                continue
            if row < 0 or row >= len(cell_boxes_2d):
                continue
            if val_col < 0 or val_col >= len(cell_boxes_2d[row]):
                continue
            cell = cell_boxes_2d[row][val_col]
            box = [
                int(cell[0] + offset_x),
                int(cell[1] + offset_y),
                int(cell[2] + offset_x),
                int(cell[3] + offset_y),
            ]
            target_list[idx]["box"] = box

    def analyze(self, image_path: str, output_dir: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        分析单张 QZ 前置画面图片。

        参数:
            image_path: 图片路径
            params: 可选，{
                "idname": "数据点号",
                "valuenameyc": "数据值",
                "valuenameyx": "数据值",
                "ycyx": "yc" / "yx"
            }

        返回:
            {
                "image_path": str,
                "image_size": [w, h],
                "status": "success" / "fail",
                "error": str,
                "params": dict,
                "data": {"qz_yc": [...], "qz_yx": [...]},  # 每条记录含 box
                "records_with_row": [(row, id, value), ...],
                "visualization": dict
            }
        """
        image_path = str(Path(image_path).resolve())
        params = self._merge_params(params)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("QZAnalyzer.analyze: %s, params=%s", image_path, params)
        analysis = analyze_qz_image(image_path, self.ocr, params)

        result = analysis["result"]
        res = {
            "image_path": analysis["image_path"],
            "image_size": analysis["image_size"],
            "status": result.get("status", "fail"),
            "error": result.get("error", ""),
            "data": result.get("Res", {"qz_yc": [], "qz_yx": []}),
        }

        out_path = output_dir / f"{Path(image_path)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        logger.info("结果已保存: %s", out_path)
        self._fill_record_boxes(result, analysis.get("records_with_row", []), analysis.get("visualization", {}))

        return res

    def analyze_batch(
        self,
        image_dir: str,
        output_dir: str,
        params: Optional[Dict[str, str]] = None,
        save_result: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量分析目录中的图片。

        参数:
            image_dir: 输入图片目录
            output_dir: 输出 JSON 目录
            params: 同 analyze
            save_result: 是否保存每个结果 JSON

        返回:
            结果字典列表
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"目录 {image_dir} 中未找到图片")

        results = []
        total = len(image_paths)
        for idx, img_path in enumerate(image_paths, start=1):
            logger.info("[%d/%d] 分析 %s", idx, total, img_path)
            try:
                res = self.analyze(str(img_path), params)
                results.append(res)
                if save_result:
                    out_path = output_dir / f"{img_path.stem}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False, indent=2)
                    logger.info("结果已保存: %s", out_path)
            except Exception as e:
                logger.exception("分析 %s 失败", img_path)
                results.append({
                    "image_path": str(img_path),
                    "status": "fail",
                    "error": str(e),
                    "data": {"qz_yc": [], "qz_yx": []},
                })
        return results

    def draw_result(
        self,
        image_path: str,
        result: Dict[str, Any],
        save_path: str,
    ) -> str:
        """
        在图片上绘制 QZ 识别结果（表格、点号列、数值列、文本框等）。

        参数:
            image_path: 原图路径
            result: analyze 返回的结果字典
            save_path: 可视化图片保存路径

        返回:
            保存路径
        """
        image_path = str(Path(image_path).resolve())
        save_path = str(Path(save_path).resolve())

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        vis = result.get("visualization", {})
        vis_image = draw_qz_annotations(
            image,
            {"Res": result.get("data", {"qz_yc": [], "qz_yx": []})},
            table_info=vis.get("table_info"),
            h_ys=vis.get("h_ys"),
            v_xs=vis.get("v_xs"),
            cell_boxes_2d=vis.get("cell_boxes_2d"),
            id_coord=vis.get("id_coord"),
            value_coord=vis.get("value_coord"),
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, vis_image)
        logger.info("QZ 可视化结果已保存: %s", save_path)
        return save_path

    def __del__(self):
        """析构时释放资源（如需）。"""
        try:
            del self.ocr
        except Exception:
            pass
