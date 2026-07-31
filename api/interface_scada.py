# -*- coding: utf-8 -*-
"""
SCADA 画面检测接口，供外部程序调用。

提供：
    - SCADAAnalyzer: 一次性加载 YOLO/数字检测模型与图元索引，
      支持单张/批量 SCADA 图片 + G 文件分析。
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# 确保项目根目录在路径中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.scada_algorithm import AlgorithmScheduler
from common.analyse_test import SimpleFileFinder
from detect_scada import process_g_file, CIMETYPE_ALGORITHM_MAP_SB, CIMETYPE_ALGORITHM_MAP_SH

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "/workspace/scadaandqz/config.json"
DEFAULT_G_DIR = "/workspace/test/scada/g"
DEFAULT_TUYUAN_DIR = "/workspace/test/tuyuan"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class SCADAAnalyzer:
    """
    SCADA 画面分析器。

    初始化时加载模型与图元文件索引，后续可多次调用 analyze 进行分析。
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        g_dir: str = DEFAULT_G_DIR,
        tuyuan_dir: str = DEFAULT_TUYUAN_DIR,
    ):
        """
        参数:
            config_path: 模型配置文件路径
            g_dir: G 文件目录
            tuyuan_dir: 图元文件目录
        """
        self.config_path = Path(config_path).resolve()
        self.g_dir = Path(g_dir).resolve()
        self.tuyuan_dir = Path(tuyuan_dir).resolve()

        if not self.g_dir.is_dir():
            raise FileNotFoundError(f"G 文件目录不存在: {self.g_dir}")
        if not self.tuyuan_dir.is_dir():
            raise FileNotFoundError(f"图元目录不存在: {self.tuyuan_dir}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.scada_model_config = self.config.get("scada_model_paths", {})
        if not self.scada_model_config:
            raise ValueError(f"配置文件 {self.config_path} 中未找到 scada_model_paths 节点")

        logger.info("SCADAAnalyzer 初始化：加载 YOLO 模型...")
        self.scheduler = AlgorithmScheduler(self.scada_model_config)
        logger.info("SCADAAnalyzer 初始化：已加载算法 %s", list(self.scheduler.algorithm_instances.keys()))

        logger.info("SCADAAnalyzer 初始化：构建图元文件索引...")
        self.file_finder = SimpleFileFinder()
        self.file_finder.add_dirs([str(self.g_dir), str(self.tuyuan_dir)])
        logger.info("SCADAAnalyzer 初始化完成")

    def _find_g_path(self, image_path: str) -> Optional[Path]:
        """
        根据图片路径在 g_dir 中查找同名 G 文件。
        支持 .jpg/.jpeg/.png/.bmp 图片。
        """
        image_path = Path(image_path)
        g_path = self.g_dir / (image_path.stem + ".g")
        if g_path.exists():
            return g_path
        return None

    def analyze(self, image_path: str, output_dir: str, g_path: Optional[str] = None) -> Dict[str, Any]:
        """
        分析单张 SCADA 图片。

        参数:
            image_path: 图片路径
            g_path: 可选 G 文件路径，默认根据图片名在 g_dir 中查找

        返回:
            {
                "image_path": str,
                "image_size": [w, h],
                "g_file": str,
                "status": "success" / "skip" / "fail",
                "error": str,
                "data": [
                    {
                        "cimeid": ..., "cimename": ..., "cimetypes": ...,
                        "ycyx": ..., "roi": [x1,y1,x2,y2], "value": ..., "status": ...
                    },
                    ...
                ]
            }
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(Path(image_path).resolve())
        if g_path is None:
            g_path = self._find_g_path(image_path)
        else:
            g_path = Path(g_path).resolve()

        if g_path is None or not g_path.exists():
            return {
                "image_path": image_path,
                "image_size": [0, 0],
                "g_file": "",
                "status": "skip",
                "error": "未找到对应 G 文件",
                "data": [],
            }

        g_path = Path(g_path)
        # 临时输出目录，仅 process_g_file 内部使用，不保存 LabelMe
        tmp_output = Path(__file__).resolve().parent / ".tmp_scada"
        tmp_output.mkdir(parents=True, exist_ok=True)

        logger.info("SCADAAnalyzer.analyze: image=%s, g=%s", image_path, g_path.name)
        result = process_g_file(
            g_path=g_path,
            image_path=Path(image_path),
            output_dir=tmp_output,
            file_finder=self.file_finder,
            scheduler=self.scheduler,
            save_labelme="",
        )
        res = {
            "image_path": result.get("image_path", image_path),
            "image_size": result.get("image_size", [0, 0]),
            "g_file": result.get("g_file", g_path.name),
            "status": result.get("status", "fail"),
            "error": result.get("message", ""),
            "data": result.get("results", []),
        }
        out_path = output_dir / f"{Path(image_path).stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        logger.info("结果已保存: %s", out_path)

        return res

    def analyze_batch(
        self,
        image_dir: str,
        output_dir: str,
        g_dir: Optional[str] = None,
        save_result: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量分析 SCADA 图片目录。

        参数:
            image_dir: 输入图片目录
            output_dir: 输出 JSON 目录
            g_dir: 可选 G 文件目录，默认使用初始化时的 g_dir
            save_result: 是否保存每个结果 JSON

        返回:
            结果字典列表
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        g_dir_path = Path(g_dir).resolve() if g_dir else self.g_dir

        image_paths = sorted(
            p for p in image_dir.glob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"目录 {image_dir} 中未找到图片")

        results = []
        total = len(image_paths)
        for idx, img_path in enumerate(image_paths, start=1):
            logger.info("[%d/%d] 分析 %s", idx, total, img_path)
            g_path = g_dir_path / (img_path.stem + ".g")
            try:
                res = self.analyze(str(img_path), str(g_path) if g_path.exists() else None)
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
                    "data": [],
                })
        return results

    def __del__(self):
        try:
            del self.scheduler
        except Exception:
            pass
