import os
import re
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_DOWNLOAD_ENABLED"] = "False"
os.environ["PADDLE_MODEL_HUB_DISABLE"] = "True"
os.environ["PADDLE_HUB_DISABLE"] = "True"
os.environ["PPX_MODEL_HUB_DISABLE"] = "True"
os.environ["PPX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLEX_DISABLE_UPDATE_CHECK"] = "True"
os.environ["PPX_DISABLE_TELEMETRY"] = "True"

import torch
torch.backends.cudnn.enabled=False
import numpy as np
from typing import List, Tuple
import os
from typing import List, Dict, Tuple, Optional, Any
import cv2
from ultralytics import YOLO
import json

from paddleocr import TextDetection
from paddleocr import TextRecognition
# ===================== 基础工具函数 =====================
def get_rect_from_poly(poly: List[List[int]]) -> List[int]:
    """将4点多边形转为矩形框 [x1,y1,x2,y2]"""
    coords = np.array(poly).reshape(-1, 2)
    return [int(np.min(coords[:, 0])), int(np.min(coords[:, 1])),
            int(np.max(coords[:, 0])), int(np.max(coords[:, 1]))]

def calculate_intersection(rect1: List[int], rect2: List[int]) -> Optional[List[int]]:
    """计算两个矩形的交集，无交集返回None"""
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    return [x1, y1, x2, y2] if (x1 < x2 and y1 < y2) else None

def is_fully_contained(inner_rect: List[int], outer_rect: List[int]) -> bool:
    """判断inner_rect是否被outer_rect完全包含"""
    return (inner_rect[0] >= outer_rect[0] and inner_rect[1] >= outer_rect[1] and
            inner_rect[2] <= outer_rect[2] and inner_rect[3] <= outer_rect[3])

def calculate_iou(rect1: List[int], rect2: List[int]) -> float:
    """计算两个矩形的IoU"""
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def remove_contained_boxes(boxes: List[List[int]]) -> List[List[int]]:
    """删除被其他框完全包含的小框（如数字0/8/6/9的内部空洞或噪点）"""
    if not boxes:
        return boxes
    # 按面积从大到小排序，优先保留大框
    sorted_boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    kept = []
    for box in sorted_boxes:
        if not any(is_fully_contained(box, other) for other in kept):
            kept.append(box)
    return kept

def merge_overlapping_boxes(boxes: List[List[int]],
                            y_overlap_thresh: float = 0.5,
                            x_gap_ratio: float = 1.0,
                            min_x_gap: int = 5,
                            iou_thresh: float = 0.15) -> List[List[int]]:
    """
    合并有显著垂直重叠或相交的文字框（处理断裂数字，如8/0被分成上下两部分）。
    按从左到右顺序，反复合并与当前基准框满足条件的框。
    """
    if not boxes:
        return boxes
    remaining = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged = []
    while remaining:
        base = list(remaining.pop(0))
        changed = True
        while changed:
            changed = False
            new_remaining = []
            for box in remaining:
                # 垂直重叠比例
                y_overlap = min(base[3], box[3]) - max(base[1], box[1])
                h_min = min(base[3]-base[1], box[3]-box[1])
                y_overlap_ratio = y_overlap / h_min if h_min > 0 else 0.0
                # 水平间距（考虑左右两种情况）
                if box[0] >= base[2]:
                    x_gap = box[0] - base[2]
                elif base[0] >= box[2]:
                    x_gap = base[0] - box[2]
                else:
                    x_gap = 0
                h_avg = ((base[3]-base[1]) + (box[3]-box[1])) / 2.0
                # IoU
                iou = calculate_iou(base, box)
                # 满足：明显垂直重叠、水平间距小 或 有交集
                if (y_overlap_ratio >= y_overlap_thresh and x_gap <= max(h_avg * x_gap_ratio, min_x_gap)) or iou >= iou_thresh:
                    base = [
                        min(base[0], box[0]),
                        min(base[1], box[1]),
                        max(base[2], box[2]),
                        max(base[3], box[3])
                    ]
                    changed = True
                else:
                    new_remaining.append(box)
            remaining = new_remaining
        merged.append(base)
    return merged

def sort_text_boxes_by_position(text_boxes: List[List[int]], row_threshold: int = 10) -> List[List[int]]:
    """按空间位置排序：先按行（上→下），同行按列（左→右）"""
    if not text_boxes:
        return []
    
    # 计算每个框的中心坐标
    boxes_with_center = []
    for box in text_boxes:
        x1, y1, x2, y2 = box
        boxes_with_center.append({
            "box": box,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2
        })
    
    # 按行聚类 + 排序
    boxes_with_center.sort(key=lambda x: x["center_y"])
    rows, current_row = [], [boxes_with_center[0]]
    for box in boxes_with_center[1:]:
        if abs(box["center_y"] - current_row[0]["center_y"]) <= row_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)
    
    # 每行内按列排序，合并结果
    sorted_boxes = []
    for row in rows:
        sorted_boxes.extend([item["box"] for item in sorted(row, key=lambda x: x["center_x"])])
    return sorted_boxes
def GetJSON(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        # 2. 解析JSON字符串为Python对象（字典/列表）
        data = json.load(f)
    return data

class TableLineToCells:
    def __init__(self, 
                 horizontal_lines_y: List[int],  # 仅横线y坐标列表（你的输入格式）
                 vertical_lines_x: List[int],   # 仅竖线x坐标列表（你的输入格式）
                 image_size: Tuple[int, int],   # 图片尺寸 (宽度, 高度)
                 line_threshold: int = 10):
        """
        初始化表格线转单元格工具（适配纯坐标列表输入）
        :param horizontal_lines_y: 横线y坐标列表 → 如 [3, 31, 61, ...]
        :param vertical_lines_x: 竖线x坐标列表 → 如 [1, 41, 448, ...]
        :param image_size: 图片尺寸 (width, height) → 用于补充边缘线
        :param line_threshold: 线坐标去重阈值（像素）
        """
        self.line_threshold = line_threshold
        self.img_height, self.img_width = image_size
        
        # 1. 补充图片边缘线的坐标
        self.full_h_ys = self._add_edge_horizontal_coords(horizontal_lines_y)
        self.full_v_xs = self._add_edge_vertical_coords(vertical_lines_x)
        
        # 2. 预处理坐标：去重、排序
        
        self.h_lines_y = self._process_horizontal_coords(self.full_h_ys)
        self.v_lines_x = self._process_vertical_coords(self.full_v_xs)
        
    def _add_edge_horizontal_coords(self, h_ys: List[int]) -> List[int]:
        """补充图片边缘的横线y坐标：顶部(y=0)、底部(y=img_height)"""
        full_h_ys = h_ys.copy()
        
        # 补充顶部边缘y=0（如果不存在）
        if 0 not in full_h_ys:
            full_h_ys.append(0)
        
        # 补充底部边缘y=img_height（如果不存在）
        if self.img_height not in full_h_ys:
            full_h_ys.append(self.img_height)
        
        return full_h_ys
    
    def _add_edge_vertical_coords(self, v_xs: List[int]) -> List[int]:
        """补充图片边缘的竖线x坐标：左侧(x=0)、右侧(x=img_width)"""
        full_v_xs = v_xs.copy()
        
        # 补充左侧边缘x=0（如果不存在）
        if 0 not in full_v_xs:
            full_v_xs.append(0)
        
        # 补充右侧边缘x=img_width（如果不存在）
        if self.img_width not in full_v_xs:
            full_v_xs.append(self.img_width)
        
        return full_v_xs
    
    def _process_horizontal_coords(self, h_ys: List[int]) -> List[int]:
        """处理横线y坐标：去重、升序排序"""
        # 去重 + 排序
        h_ys = sorted(list(set(h_ys)))
        
        # 过滤过近的坐标（差值小于阈值视为同一条线）
        filtered_y = []
        for y in h_ys:
            if not filtered_y or abs(y - filtered_y[-1]) > self.line_threshold:
                filtered_y.append(y)

        if len(filtered_y) >= 4:
            base_diff = filtered_y[-3] - filtered_y[-4]
            # print(f"base_diff:{base_diff}")
            last_diff = filtered_y[-1] - filtered_y[-2]
            # print(f"last_diff:{last_diff}")
            if last_diff <= base_diff * 0.8:
                second_last_diff = filtered_y[-2] - filtered_y[-3]
                # print(f"second_last_diff:{second_last_diff}")
                if second_last_diff > base_diff * 0.8:
                    filtered_y.pop()
                else:
                    filtered_y.pop()
                    filtered_y.pop()

        return filtered_y
    
    def _process_vertical_coords(self, v_xs: List[int]) -> List[int]:
        """处理竖线x坐标：去重、升序排序"""
        # 去重 + 排序
        v_xs = sorted(list(set(v_xs)))
        
        # 过滤过近的坐标（差值小于阈值视为同一条线）
        filtered_x = []
        for x in v_xs:
            if not filtered_x or abs(x - filtered_x[-1]) > self.line_threshold:
                filtered_x.append(x)
        return filtered_x
    
    def generate_cell_boxes_2d(self) -> List[List[List[int]]]:
        """
        生成二维单元格坐标（核心函数）
        :return: CELL_BOXES_2D 格式的二维列表
                 [[[x1,y1,x2,y2], ...], ...]
        """
        cell_boxes_2d = []
        
        # 遍历每两条相邻横线（行）
        
        for row_idx in range(len(self.h_lines_y) - 1):
            y_top = self.h_lines_y[row_idx]    # 行的上边界
            y_bottom = self.h_lines_y[row_idx + 1]  # 行的下边界
            row_cells = []
            
            # 遍历每两条相邻竖线（列）
            for col_idx in range(len(self.v_lines_x) - 1):
                x_left = self.v_lines_x[col_idx]  # 列的左边界
                x_right = self.v_lines_x[col_idx + 1]  # 列的右边界
                
                # 单元格坐标：[x1, y1, x2, y2]
                cell_rect = [x_left, y_top, x_right, y_bottom]
                row_cells.append(cell_rect)
            
            cell_boxes_2d.append(row_cells)
        
        return cell_boxes_2d

class qyboxmodel():
    def __init__(self,model_path):
        self.conf_threshold = 0.1
        self.det_model = self._load_yolo_model(model_path)
    def _load_yolo_model(self, model_path: str) -> YOLO:
        """加载YOLO11表格检测模型"""
        try:
            model = YOLO(model_path)  # 支持本地.pt文件或官方模型（如yolo11n.pt）
            print(f"YOLO11表格检测模型加载成功: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"加载YOLO11模型失败: {str(e)}")

    def predict(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测画面中置信度最高的表格
        :param image: cv2读取的图像数组（BGR格式）
        :return: 最高置信度表格的信息，格式：
                {
                    "conf": 置信度,
                    "bbox": [x1, y1, x2, y2],  # 表格左上角/右下角坐标
                    "center": [cx, cy],        # 表格中心点
                    "width": 宽度,
                    "height": 高度
                }
                无表格时返回None
        """
        # YOLO11推理
        results = self.det_model(image, conf=self.conf_threshold)
        
        # 提取表格检测结果（假设表格的类别id是0，可根据你的模型调整）
        table_boxes = []
        for r in results:
            for box in r.boxes:
                if box.conf >= self.conf_threshold:
                    # 转换为xyxy格式的坐标（x1,y1,x2,y2）
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    table_boxes.append([x1, y1, x2, y2])
        return table_boxes
        
class Image2TableIamge():
    def __init__(self,model_path):
        self.conf_threshold = 0.1
        self.table_det_model = self._load_yolo_model(model_path)
    def _load_yolo_model(self, model_path: str) -> YOLO:
        """加载YOLO11表格检测模型"""
        try:
            model = YOLO(model_path)  # 支持本地.pt文件或官方模型（如yolo11n.pt）
            print(f"YOLO11表格检测模型加载成功: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"加载YOLO11模型失败: {str(e)}")

    def detect_best_table(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测画面中置信度最高的表格
        :param image: cv2读取的图像数组（BGR格式）
        :return: 最高置信度表格的信息，格式：
                {
                    "conf": 置信度,
                    "bbox": [x1, y1, x2, y2],  # 表格左上角/右下角坐标
                    "center": [cx, cy],        # 表格中心点
                    "width": 宽度,
                    "height": 高度
                }
                无表格时返回None
        """
        # YOLO11推理
        results = self.table_det_model(image, conf=self.conf_threshold)
        
        # 提取表格检测结果（假设表格的类别id是0，可根据你的模型调整）
        table_boxes = []
        for r in results:
            for box in r.boxes:
                if box.conf >= self.conf_threshold:
                    # 转换为xyxy格式的坐标（x1,y1,x2,y2）
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    table_boxes.append({
                        "conf": conf,
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1+x2)//2, (y1+y2)//2],
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
        
        # 筛选置信度最高的表格
        if not table_boxes:
            print("未检测到符合阈值的表格")
            return None
        
        best_table = max(table_boxes, key=lambda x: x["conf"])
        # print(f"检测到最高置信度表格: 置信度={best_table['conf']:.3f}, 位置={best_table['bbox']}")
        return best_table
        
    def crop_table_image(self, image: np.ndarray, table_info: Dict) -> np.ndarray:
        """
        根据表格检测结果裁剪出表格区域
        :param image: 原始图像
        :param table_info: detect_best_table返回的表格信息
        :return: 裁剪后的表格图像
        """
        x1, y1, x2, y2 = table_info["bbox"]
        # 防止坐标越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        return image[y1:y2, x1:x2]

    def preprocess_image2(self,table_img, 
                        row_threshold=0.8,  # 行有效占比阈值（80%）
                        col_threshold=0.8,  # 列有效占比阈值（80%）
                        nms_window=10      # NMS窗口大小（相邻5行/列）
                        ):
        """
        提取表格水平线和垂直线：
        1. 统计每行/列白色像素占比，筛选有效区域
        2. 类NMS算法保留相邻区域中最优线
        """
        img = table_img.copy()
        img_height, img_width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 自适应二值化（增强表格线对比度）
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        # 形态学操作（强化表格线，去除噪点）
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 反相二值化（表格线为白色，背景为黑色）
        thresh_inv = cv2.bitwise_not(thresh)  # 白色：255（表格线），黑色：0（背景）

        # ===================== 1. 提取水平线（按行统计）=====================
        # 统计每行的白色像素占比（行高为3像素，避免单像素误差）
        row_white_ratio = []
        for y in range(img_height):
            # 取y-1/y/y+1三行的平均值，增强鲁棒性
            start_y = max(0, y-1)
            end_y = min(img_height, y)
            row_region = thresh_inv[start_y:end_y, :]
            white_pixels = np.sum(row_region == 255)
            total_pixels = row_region.size
            ratio = white_pixels / total_pixels if total_pixels > 0 else 0
            row_white_ratio.append(ratio)
        
        # 筛选有效行（占比≥阈值）
        valid_rows = [y for y, ratio in enumerate(row_white_ratio) if ratio >= row_threshold]
        
        # 类NMS：相邻nms_window行中保留占比最大的行
        h_ys = []
        if valid_rows:
            valid_rows_sorted = sorted(valid_rows)
            i = 0
            while i < len(valid_rows_sorted):
                # 取当前窗口内的所有行
                window_end = i
                while window_end < len(valid_rows_sorted) and valid_rows_sorted[window_end] - valid_rows_sorted[i] < nms_window:
                    window_end += 1
                window_rows = valid_rows_sorted[i:window_end]
                
                # 找窗口内占比最大的行
                max_ratio = -1
                best_y = window_rows[0]
                for y in window_rows:
                    if row_white_ratio[y] > max_ratio:
                        max_ratio = row_white_ratio[y]
                        best_y = y
                
                h_ys.append(best_y)
                i = window_end

        # ===================== 2. 提取垂直线（按列统计）=====================
        # 统计每列的白色像素占比（列宽为3像素，避免单像素误差）
        col_white_ratio = []
        for x in range(img_width):
            # 取x-1/x/x+1三列的平均值，增强鲁棒性
            start_x = max(0, x-1)
            end_x = min(img_width, x)
            col_region = thresh_inv[:, start_x:end_x]
            white_pixels = np.sum(col_region == 255)
            total_pixels = col_region.size
            ratio = white_pixels / total_pixels if total_pixels > 0 else 0
            col_white_ratio.append(ratio)
        
        # 筛选有效列（占比≥阈值）
        valid_cols = [x for x, ratio in enumerate(col_white_ratio) if ratio >= col_threshold]
        
        # 类NMS：相邻nms_window列中保留占比最大的列
        v_xs = []
        if valid_cols:
            valid_cols_sorted = sorted(valid_cols)
            i = 0
            while i < len(valid_cols_sorted):
                # 取当前窗口内的所有列
                window_end = i
                while window_end < len(valid_cols_sorted) and valid_cols_sorted[window_end] - valid_cols_sorted[i] < nms_window:
                    window_end += 1
                window_cols = valid_cols_sorted[i:window_end]
                
                # 找窗口内占比最大的列
                max_ratio = -1
                best_x = window_cols[0]
                for x in window_cols:
                    if col_white_ratio[x] > max_ratio:
                        max_ratio = col_white_ratio[x]
                        best_x = x
                
                v_xs.append(best_x)
                i = window_end
        h_ys = sorted(h_ys)
        v_xs = sorted(v_xs)
        def filter_close_lines(lines, min_gap=nms_window):
            # 确保相邻两条线的间距至少为 nms_window，把过于贴近的线再次剔除
            filtered = []
            prev = -min_gap
            for line in lines:
                if line - prev >= min_gap:
                    filtered.append(line)
                    prev = line
            return filtered
        h_ys = filter_close_lines(h_ys)
        v_xs = filter_close_lines(v_xs)
        # 最终排序
        return h_ys, v_xs

    def run(self,image):
        """
        提取表格区域和表格线
        h_ys: 表格横线列表
        v_xs: 表格竖线列表
        smallimage: 表格区域
        """
        best_table=self.detect_best_table(image)
        if(best_table):
            smallimage=self.crop_table_image(image,best_table)
            h_ys,v_xs=self.preprocess_image2(smallimage)
            return h_ys,v_xs,smallimage
        else:
            return None,None,None

class Table:
    def __init__(self, img, cell_boxes_2d,
                 text_polys: List[List[List[int]]], text_bcboxes:List[List[int]]=[],row_threshold: int = 10):
        """
        初始化表格OCR引擎
        :param image_path: 表格图片路径
        :param cell_boxes_2d: 二维单元格坐标 [[[x1,y1,x2,y2], ...], ...]
        :param text_polys: PaddleOCR检测的文字框多边形列表
        :param row_threshold: 行聚类阈值（像素）
        """
        self.cell_boxes_2d = cell_boxes_2d
        self.text_rects = [get_rect_from_poly(poly) for poly in text_polys]  # 文字框转矩形
        self.textbc_rects=text_bcboxes
        self.row_threshold = row_threshold
        self.img = img
        # 核心映射关系
        self.cell_text_mapping = self._match_text_to_cells()  # (行,列) → [文字框列表]
        self.textbox_cell_mapping = {}  # 文字框tuple → (行,列)：反向映射，用于推理结果回绑
        for cell_key, text_boxes in self.cell_text_mapping.items():
            for box in text_boxes:
                self.textbox_cell_mapping[tuple(box)] = cell_key
        
        # 缓存：文字框tuple → 裁剪后的子图
        self.textbox_subimages = {}
        # 缓存：目标单元格 → 对应的文字框列表（用于最终拼接）
        self.target_cell_textboxes = {}

    def _match_text_to_cells(self) -> Dict[Tuple[int, int], List[List[int]]]:
        """匹配文字框到对应的单元格（核心匹配逻辑）"""
        cell_text_mapping = {}
        for row_idx, row_cells in enumerate(self.cell_boxes_2d):
            for col_idx, cell_rect in enumerate(row_cells):
                cell_key = (row_idx, col_idx)
                cell_text_mapping[cell_key] = []
                cell_height = cell_rect[3] - cell_rect[1]
                min_valid_h = cell_height * 0.3
                min_valid_w = cell_height * 0.3

                # 遍历所有文字框，匹配当前单元格
                for text_rect in self.text_rects:
                    intersection = calculate_intersection(text_rect, cell_rect)
                    if not intersection:
                        continue
                    
                    inter_h = intersection[3] - intersection[1]
                    inter_w = intersection[2] - intersection[0]
                    fully_contained = is_fully_contained(text_rect, cell_rect)

                    # 应用筛选规则
                    if fully_contained:
                        cell_text_mapping[cell_key].append(text_rect)
                    elif inter_h >= min_valid_h and inter_w >= min_valid_w:
                        cell_text_mapping[cell_key].append(intersection)
        for row_idx, row_cells in enumerate(self.cell_boxes_2d):
            for col_idx, cell_rect in enumerate(row_cells):
                cell_key = (row_idx, col_idx)
                # if (cell_text_mapping[cell_key] !=[]):
                #     continue
                cell_height = cell_rect[3] - cell_rect[1]
                min_valid_h = cell_height * 0.3
                min_valid_w = cell_height * 0.2

                # 遍历所有文字框，匹配当前单元格
                for text_rect in self.textbc_rects:
                    intersection = calculate_intersection(text_rect, cell_rect)
                    if not intersection:
                        continue
                    
                    inter_h = intersection[3] - intersection[1]
                    inter_w = intersection[2] - intersection[0]
                    fully_contained = is_fully_contained(text_rect, cell_rect)
                    if (cell_text_mapping[cell_key] ==[]):
                        # 应用筛选规则
                        if fully_contained:
                            cell_text_mapping[cell_key].append(text_rect)
                        elif inter_h >= min_valid_h and inter_w >= min_valid_w:
                            cell_text_mapping[cell_key].append(intersection)
                    else:
                        # 应用筛选规则
                        if fully_contained:
                            lastrect=text_rect
                        elif inter_h >= min_valid_h and inter_w >= min_valid_w:
                            lastrect=intersection
                        else:
                            lastrect=[]
                        if(lastrect!=[]):
                            for n in range(len(cell_text_mapping[cell_key])):
                                onbox=cell_text_mapping[cell_key][n]
                                onboxsize=(onbox[1]-onbox[0])*(onbox[3]-onbox[2])
                                lastrectsize=(lastrect[1]-lastrect[0])*(lastrect[3]-lastrect[2])
                                # print(lastrect)
                                if(lastrectsize>onboxsize):
                                    cell_text_mapping[cell_key]=[]
                                    cell_text_mapping[cell_key].append(lastrect)
                                    break
        return cell_text_mapping

    def _detect_text_regions_in_cell(self, cell_roi: np.ndarray, cell_box: List[int]) -> List[List[int]]:
        """
        使用传统CV算法检测单元格内的多个文字区域。
        对多种背景色（白/绿/黄）和文字色（黑/红等）鲁棒。
        返回原图绝对坐标的 text_box 列表 [[x1,y1,x2,y2], ...]，未检测到返回空列表。
        """
        h, w = cell_roi.shape[:2]
        if h < 3 or w < 3:
            return []

        gray = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY) if len(cell_roi.shape) == 3 else cell_roi.copy()

        # 去掉边缘2px，避免单元格边框干扰
        border = 2
        if h > border * 2 and w > border * 2:
            inner = gray[border:h - border, border:w - border]
        else:
            inner = gray
            border = 0
        ih, iw = inner.shape

        # 尝试多种二值化/检测方法
        candidates = []
        _, bin1 = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates.append(("otsu_inv", bin1))
        _, bin2 = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(("otsu", bin2))

        bs = max(3, min(iw, ih, 11))
        if bs % 2 == 0:
            bs += 1
        try:
            bin3 = cv2.adaptiveThreshold(inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, bs, 2)
            candidates.append(("adaptive", bin3))
        except Exception:
            pass

        sobelx = cv2.Sobel(inner, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(inner, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255))
        _, bin4 = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(("grad", bin4))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        total_area = ih * iw
        best_boxes = []
        best_score = -1.0

        for name, binary in candidates:
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 连通域分析：获取多个独立文字区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
            regions = []
            for i in range(1, num_labels):
                x, y, bw, bh, area = stats[i]
                if area < 5 or area > total_area * 0.8:
                    continue
                if bw < 5 or bh < 3:
                    continue
                if bw > iw * 0.95 or bh > ih * 0.95:
                    continue
                # 过滤极小噪声：面积 < 1% cell 面积 且 宽高均 < 8px
                if area < total_area * 0.01 and bw < 8 and bh < 8:
                    continue
                # 过滤扁平线状噪声（如下划线、横线），但保留位于单元格垂直中心区域的负号
                if bh <= 3 and bw > bh * 3:
                    region_cy = y + bh / 2
                    center_ratio = region_cy / ih if ih > 0 else 0.5
                    # 负号通常位于单元格垂直中心；下划线/横线通常在底部或顶部
                    if not (0.25 <= center_ratio <= 0.75):
                        continue
                # 过滤宽高比例过大的
                # if max(bw, bh) / max(min(bw, bh), 1) > 2:
                #     continue
                regions.append((x, y, x + bw, y + bh))

            if not regions:
                continue

            # 按从上到下、从左到右排序
            regions = sorted(regions, key=lambda r: (r[1], r[0]))
            # 移除被大框完全包含的小框（数字0/8/6/9的内部空洞、噪点）
            regions = remove_contained_boxes([list(r) for r in regions])
            # 合并同一行上相邻或断裂的文字块
            merged = merge_overlapping_boxes(regions, y_overlap_thresh=0.45, x_gap_ratio=1.5, min_x_gap=5)

            # 转回原图绝对坐标
            abs_boxes = []
            for r in merged:
                rtx1, rty1, rtx2, rty2 = r
                if border > 0:
                    rtx1 += border
                    rty1 += border
                    rtx2 += border
                    rty2 += border
                abs_boxes.append([
                    cell_box[0] + rtx1,
                    cell_box[1] + rty1,
                    cell_box[0] + rtx2,
                    cell_box[1] + rty2,
                ])

            # 评分：文字区域总面积占比适中
            text_area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in merged)
            pixel_ratio = text_area / total_area
            if 0.02 <= pixel_ratio <= 0.7:
                score = pixel_ratio
                if name == "grad":
                    score *= 1.15
                if score > best_score:
                    best_score = score
                    best_boxes = abs_boxes

        return best_boxes

    def crop_textbox_subimages(self, target_cells: List[Tuple[int, int]],
                               save_dir: str = "/workspace/scadaandqz/textbox_subimages", qz:bool =False) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        裁剪目标单元格内所有文字框的小图。
        全部使用传统CV算法检测文字区域，不再依赖 TextDetModel 的结果。
        :param target_cells: 目标单元格 [(行,列), ...]
        :param save_dir: 子图保存目录（可选）
        :param qz: 是否为数值列模式；True 时一个单元格只保留一个合并后的文字框
        :return: (文字框子图列表, 文字框对应的单元格key列表) → 两个列表顺序完全一致
        """
        os.makedirs(save_dir, exist_ok=True)
        subimages = []
        textbox_cell_keys = []

        for cell_key in target_cells:
            row_idx, col_idx = cell_key
            if (row_idx >= len(self.cell_boxes_2d)) or (col_idx >= len(self.cell_boxes_2d[row_idx])):
                raise ValueError(f"单元格({row_idx},{col_idx})超出范围")

            # 获取单元格在原图上的坐标
            cell_box = self.cell_boxes_2d[row_idx][col_idx]
            cx1, cy1, cx2, cy2 = cell_box
            img_h, img_w = self.img.shape[:2]

            cell_roi = self.img[cy1:cy2, cx1:cx2]
            if cell_roi.size == 0:
                continue

            # 全部使用传统算法检测文字区域（支持一个单元格内多个文字块）
            text_boxes = self._detect_text_regions_in_cell(cell_roi, cell_box)

            if not text_boxes:
                # 兜底：裁剪整个 cell（带 padding）
                pad_x = min(5, max(2, (cx2 - cx1) // 10))
                pad_y = min(5, max(2, (cy2 - cy1) // 10))
                fx1 = max(0, cx1 + pad_x)
                fy1 = max(0, cy1 + pad_y)
                fx2 = min(img_w, cx2 - pad_x)
                fy2 = min(img_h, cy2 - pad_y)
                if fx2 <= fx1 or fy2 <= fy1:
                    fx1, fy1, fx2, fy2 = cx1, cy1, cx2, cy2  # 直接用原单元格
                text_boxes = [[fx1, fy1, fx2, fy2]]

            # qz 模式：每个单元格只有一个数值，合并剩余文字框为单个框
            if qz and len(text_boxes) > 1:
                text_boxes = [[min(b[0] for b in text_boxes),
                               min(b[1] for b in text_boxes),
                               max(b[2] for b in text_boxes),
                               max(b[3] for b in text_boxes)]]

            self.target_cell_textboxes[cell_key] = text_boxes

            for box_idx, text_box in enumerate(text_boxes):
                x1, y1, x2, y2 = text_box
                y1 = max(0, y1-2)
                x1 = max(0, x1-4)
                y2 = min(img_h, y2+2)
                x2 = min(img_w, x2+2)
                subimg = self.img[y1:y2, x1:x2].copy()

                # save_path = os.path.join(save_dir, f"cell_{row_idx}_{col_idx}_textbox_{box_idx}.png")
                # cv2.imwrite(save_path, subimg)

                subimages.append(subimg)
                textbox_cell_keys.append(cell_key)

        return subimages, textbox_cell_keys

    def concat_ocr_results(self, target_cells: List[Tuple[int, int]], 
                           ocr_results: List[str], textbox_cell_keys: List[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
        """
        将文字框的Batch推理结果回绑并拼接成单元格级别的字符串
        :param target_cells: 目标单元格 [(行,列), ...]
        :param ocr_results: 文字框OCR推理结果列表（顺序与crop返回的subimages一致）
        :param textbox_cell_keys: crop返回的文字框对应单元格key列表
        :return: {(行,列): 最终拼接字符串, ...}
        """
        # 第一步：构建「单元格→文字框→识别文字」的映射
        cell_textbox_text = {}
        for cell_key in target_cells:
            cell_textbox_text[cell_key] = {}  # 初始化：cell_key → {文字框tuple: 识别文字}
        
        # 遍历推理结果，回绑到对应的文字框和单元格
        for idx, ocr_text in enumerate(ocr_results):
            if idx >= len(textbox_cell_keys):
                break
            cell_key = textbox_cell_keys[idx]
            # 获取当前推理结果对应的文字框（按顺序匹配）
            text_boxes = self.target_cell_textboxes.get(cell_key, [])
            # 找到当前idx对应的文字框（计算偏移量）
            offset = 0
            for ck in target_cells:
                if ck == cell_key:
                    break
                offset += len(self.target_cell_textboxes.get(ck, []))
            box_idx = idx - offset
            if box_idx < len(text_boxes):
                text_box = text_boxes[box_idx]
                cell_textbox_text[cell_key][tuple(text_box)] = ocr_text
        
        # 第二步：按单元格排序并拼接字符串
        final_results = {}
        for cell_key in target_cells:
            text_boxes = self.target_cell_textboxes.get(cell_key, [])
            sorted_boxes = sort_text_boxes_by_position(text_boxes, self.row_threshold)
            
            # 按排序后的文字框拼接文字
            concat_str = ""
            for box in sorted_boxes:
                concat_str += cell_textbox_text[cell_key].get(tuple(box), "")
            final_results[cell_key] = concat_str
        
        return final_results

def get_point_and_value_coords(
    ta, cell_boxes_2d, TextRecModel, max_check_rows=3, IDname=None, ValueName=None
):
    """
    逐行识别表格（最多3行），仅返回「点号」和「遥测值」的坐标，找到即停。
    IDname 和 ValueName 支持传入字符串或字符串列表，只要匹配到其中一个即可。
    
    Args:
        ta (Table): 已初始化的Table对象
        cell_boxes_2d (list): 单元格二维坐标
        TextRecModel: 文本识别模型实例
        max_check_rows (int): 最大检查行数（默认3行）
        IDname (str/list): 点号列匹配名称，可为单个字符串或多个字符串列表
        ValueName (str/list): 值列匹配名称，可为单个字符串或多个字符串列表
    
    Returns:
        tuple: (point_coord, value_coord)
            - point_coord: 点号坐标 (行, 列)，未找到为None
            - value_coord: 遥测值坐标 (行, 列)，未找到为None
    """
    if IDname is None:
        IDname = ["点号"]
    if ValueName is None:
        ValueName = ["遥测值"]
    
    # 统一转换为列表处理
    if isinstance(IDname, str):
        IDname = [IDname]
    if isinstance(ValueName, str):
        ValueName = [ValueName]
    
    matched_id_name = None
    matched_value_name = None
    point_coord = None   # 点号坐标
    value_coord = None   # 遥测值坐标
    
    # 遍历指定行数，找到两个坐标后立即终止
    total_rows = min(max_check_rows, len(cell_boxes_2d))
    for row_idx in range(total_rows):
        # 1. 处理当前行
        row_cols = len(cell_boxes_2d[row_idx])
        target_cells = [(row_idx, col_idx) for col_idx in range(row_cols)]
        
        # 2. 裁剪+识别当前行文本
        subimages, textbox_cell_keys = ta.crop_textbox_subimages(target_cells)
        if not subimages:
            continue
        reses = TextRecModel.predict(subimages, batch_size=16)
        rec_textses = [res["rec_text"] for res in reses]
        
        # 3. 合并当前行结果并查找目标
        row_labels = ta.concat_ocr_results(target_cells, rec_textses, textbox_cell_keys)
        for (row, col), text in row_labels.items():
            # 找到点号（列表中任意一个匹配即可）
            if point_coord is None:
                for name in IDname:
                    if name in text:
                        point_coord = (row, col)
                        matched_id_name = name
                        print(f"找到{name} → 坐标({row},{col})")
                        break
            
            # 找到遥测值（列表中任意一个匹配即可）
            if value_coord is None:
                for name in ValueName:
                    if name in text:
                        value_coord = (row, col)
                        matched_value_name = name
                        print(f"找到{name} → 坐标({row},{col})")
                        break
        
        # 4. 两个坐标都找到，立即终止（核心优化）
        if point_coord is not None and value_coord is not None:
            print(f"{matched_id_name}和{matched_value_name}都已找到，提前终止识别")
            break
    
    # 未找到的提示
    if point_coord is None:
        print("警告：前{}行未找到「{}」".format(max_check_rows, IDname))
    if value_coord is None:
        print("警告：前{}行未找到「{}」".format(max_check_rows, ValueName))
    
    # 仅返回两个坐标，无其他冗余信息
    return point_coord, value_coord
def crop_same_col_below(ta,cell_boxes_2d, coord,TextRecModel,qz=False):
    """
    裁剪指定坐标下方同列的所有单元格图片，并保留crop_textbox_subimages完整返回值
    Args:
        cell_boxes_2d (list): 单元格二维坐标
        coord (tuple): 参考坐标 (行, 列)
    Returns:
        list: 同列下方的单元格数据列表，每个元素为 (subimage, textbox_cell_keys)
    """
    if coord is None:
        print("参考坐标为空，无法裁剪")
        return []
    
    ref_row, ref_col = coord
    target_cells = []  
    
    # 遍历参考行下方的所有行
    for row_idx in range(ref_row + 1, len(cell_boxes_2d)):
        # 确保列索引不越界
        if ref_col >= len(cell_boxes_2d[row_idx]):
            continue
        
        # 生成当前单元格坐标 (行, 列)
        target_cell = (row_idx, ref_col)
        target_cells.append(target_cell)

    subimages, textbox_cell_keys = ta.crop_textbox_subimages(target_cells,qz=qz) 
    reses = TextRecModel.predict(subimages, batch_size=16)
    rec_textses = [res["rec_text"] for res in reses]
    row_labels = ta.concat_ocr_results(target_cells, rec_textses, textbox_cell_keys)  

    return row_labels
def keep_only_number_chars(input_str: str) -> str:
    """
    清理字符串：仅保留合法数字格式
    ✅ 负号只能在开头
    ✅ 小数点不能在开头
    ✅ 最多一个小数点
    ✅ 只保留 数字、-、.
    """
    # 处理 None、非字符串
    if input_str is None:
        return ""
    if not isinstance(input_str, str):
        input_str = str(input_str)

    # 第一步：只保留数字、负号、小数点
    cleaned = re.sub(r'[^-0-9.]', '', input_str.strip())

    if not cleaned:
        return ""

    # 第二步：负号只能保留第一个，其余全部删除
    if cleaned.count('-') > 1:
        cleaned = '-' + cleaned.replace('-', '')[1:]

    # 第三步：确保负号只在第一位
    if '-' in cleaned and cleaned[0] != '-':
        cleaned = cleaned.replace('-', '')

    # 第四步：小数点不能在第一位，且只保留一个
    if cleaned.startswith('.'):
        cleaned = cleaned[1:]  # 去掉开头的小数点
    if cleaned.count('.') > 1:
        parts = cleaned.split('.', 1)
        cleaned = parts[0] + '.' + parts[1].replace('.', '')

    return cleaned
def match_id_and_value(id_dict, value_dict,point_coord, value_coord):
    """
    按行匹配ID和Value，过滤空值，生成[{id:..., value:...}]格式数组
    Args:
        id_dict: ID列字典 {(行, 列): 值}
        value_dict: Value列字典 {(行, 列): 值}
    Returns:
        list: 匹配后的数组，元素为{"id": 行值, "value": 行值}
    """
    result_list = []
    
    # 提取所有行号（从ID字典中获取，确保行号完整）
    row_numbers = set([k[0] for k in id_dict.keys()])
    _,lieid=point_coord
    _,lievalue=value_coord
    # 按行号升序遍历（保证顺序）
    for row in sorted(row_numbers):
        # 1. 查找当前行的ID值（列固定为1）
        id_key = (row, lieid)
        id_value = id_dict.get(id_key, '').strip()
        id_value = re.sub(r'[^0-9]', '', id_value)
        
        # 2. 查找当前行的Value值（列固定为5）
        value_key = (row, lievalue)
        value_value = value_dict.get(value_key, '').strip()
        print(id_value, value_value)
        if value_value and value_value.lower() in {"o", "c", "n"}:
            value_value = "0"
        
        value_value = keep_only_number_chars(value_value)
        if value_value == "":
            value_value = "0"
        
        # 3. 过滤空值：ID和Value都不为空才保留
        if id_value and value_value:
            result_list.append({
                "id": id_value,
                "value": value_value
            })
    
    return result_list
def visualize_table_lines(table_img, h_ys, v_xs, save_path="table_lines_visualized.jpg"):
    """
    可视化表格线：在原图上绘制检测到的水平线和垂直线
    :param table_img: 原始表格图片
    :param h_ys: 水平线的y坐标列表
    :param v_xs: 垂直线的x坐标列表
    :param save_path: 可视化结果保存路径
    :return: 绘制后的图片
    """
    # 创建原图副本用于绘制（避免修改原图）
    img_visual = table_img.copy()
    img_height, img_width = img_visual.shape[:2]
    
    
    # 绘制水平线（红色，线宽2）
    for y in h_ys:
        cv2.line(
            img_visual, 
            (0, y), (img_width, y),  # 从左到右画满整行
            (0, 0, 255),             # 红色 (BGR格式)
            thickness=2              # 线宽，可根据需要调整
        )
        # 标注y坐标（方便调试）
        cv2.putText(
            img_visual, 
            f"y={y}", 
            (10, y-5),               # 文字位置（在线上方）
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,                     # 字体大小
            (0, 0, 255),             # 红色
            1                        # 文字线宽
        )
    
    # 绘制垂直线（绿色，线宽2）
    for x in v_xs:
        cv2.line(
            img_visual, 
            (x, 0), (x, img_height),  # 从上到下画满整列
            (0, 255, 0),              # 绿色 (BGR格式)
            thickness=2               # 线宽，可根据需要调整
        )
        # 标注x坐标（方便调试）
        cv2.putText(
            img_visual, 
            f"x={x}", 
            (x+5, 20),                # 文字位置（在线右侧）
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,                      # 字体大小
            (0, 255, 0),              # 绿色
            1                         # 文字线宽
        )
    # 保存可视化结果
    cv2.imwrite(save_path, img_visual)  
    return img_visual


class OCRRecognizer:
    def __init__(self,scada_model_config):
        TextRecModelpath=scada_model_config["TextRecModelpath"]
        TextDetModelpath=scada_model_config["TextDetModelpath"]
        tabelmodelpath=scada_model_config["tabelmodelpath"]
        wzqymodelpath=scada_model_config["wzqymodelpath"]
        self.TextRecModel=TextRecognition(model_name="PP-OCRv5_mobile_rec",model_dir=TextRecModelpath)
        self.TextDetModel = TextDetection(model_name="PP-OCRv5_server_det",model_dir=TextDetModelpath)    
        self.tabelmodel=Image2TableIamge(tabelmodelpath)
        self.wzqymodel=qyboxmodel(wzqymodelpath)

    # def run(self, frame: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    #     h_ys,v_xs,table_img=self.tabelmodel.run(frame)
    #     ValueNameyc=params["valuenameyc"]
    #     ValueNameyx=params["valuenameyx"]
    #     ycyx=params["ycyx"]
    #     if(ycyx=="yc"):
    #         ValueName=ValueNameyc
    #     else:
    #         ValueName=ValueNameyx
    #     IDname=params["idname"]
    #     if(h_ys):
    #         text_polys=self.TextDetModel.predict(table_img)[0]["dt_polys"]
    #         text_boxs=self.wzqymodel.predict(table_img)
    #         tabelcline=TableLineToCells(h_ys,v_xs,table_img.shape[:2])
    #         cell_boxes_2d = tabelcline.generate_cell_boxes_2d()
    #         ta=Table(table_img,cell_boxes_2d,text_polys,text_boxs)
    #         ID_coord, value_coord=get_point_and_value_coords(ta, cell_boxes_2d, self.TextRecModel, max_check_rows=3,IDname=IDname,ValueName=ValueName)
    #         if(ID_coord is None or value_coord is None):
    #             return {
    #             "Res": {
    #                 "qz_yc":[],
    #                 "qz_yx":[]
    #             },
    #             "status": "fail",
    #             "error":"没有找到标头"+IDname+" "+ValueName 
    #             }
    #         ID_labels=crop_same_col_below(ta,cell_boxes_2d, ID_coord,self.TextRecModel,True)
    #         value_labels=crop_same_col_below(ta,cell_boxes_2d, value_coord,self.TextRecModel,True)
    #         reslist=match_id_and_value(ID_labels, value_labels,ID_coord, value_coord)
    #         res={"qz_yc":[],
    #                 "qz_yx":[]}
    #         if(ycyx=="yc"):
    #             res["qz_yc"]=reslist
    #         else:
    #             res["qz_yx"]=reslist     
    #         return {
    #             "Res": res,
    #             "status": "success",
    #             "error":""
    #         }
    #     else:
    #         return {
    #             "Res":{"qz_yc":[],
    #                 "qz_yx":[]},
    #             "status": "fail",
    #             "error":"没有找到表"
    #         }
        
    def run(self, frame: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        ValueNameyc = params["valuenameyc"]
        ValueNameyx = params["valuenameyx"]
        ycyx = params["ycyx"]

        vis_info = {
            "table_info": None,
            "h_ys": None,
            "v_xs": None,
            "cell_boxes_2d": None,
            "id_coord": None,
            "value_coord": None
        }

        # 新增：先检测表格位置，同时拿到 table_box
        best_table = self.tabelmodel.detect_best_table(frame)
        
        if not best_table:
            return {
                "Res": {"qz_yc": [], "qz_yx": []},
                "status": "fail",
                "error": "未检测到表格",
                "visualization": vis_info
            }

        # Step 1: 表格检测
        table_img = self.tabelmodel.crop_table_image(frame, best_table)
        # cv2.imwrite("table.jpg", table_img)
        h_ys, v_xs = self.tabelmodel.preprocess_image2(table_img)
        vis_info["table_info"] = best_table
        vis_info["h_ys"] = h_ys
        vis_info["v_xs"] = v_xs
        if h_ys and v_xs and table_img is not None:
            # Step 2: 文字检测 & 区域检测
            text_polys=self.TextDetModel.predict(table_img)[0]["dt_polys"]
            text_boxs=self.wzqymodel.predict(table_img)
            # Step 3: 生成单元格
            tabelcline=TableLineToCells(h_ys,v_xs,table_img.shape[:2])
            cell_boxes_2d = tabelcline.generate_cell_boxes_2d()
            # Step 4: 构建 Table 对象 & 识别表头
            ta = Table(table_img, cell_boxes_2d, text_polys, text_boxs)

            IDname = ValueNameyc if ycyx == "yc" else ValueNameyx
            ValueName = ValueNameyc if ycyx == "yc" else ValueNameyx

            ID_coord, value_coord = get_point_and_value_coords(
                ta, cell_boxes_2d, self.TextRecModel, max_check_rows=3,
                IDname=IDname, ValueName=ValueName
            )
            vis_info["id_coord"] = ID_coord
            vis_info["value_coord"] = value_coord
            if ID_coord is None or value_coord is None:
                id_label = IDname[0] if isinstance(IDname, list) and IDname else IDname
                value_label = ValueName[0] if isinstance(ValueName, list) and ValueName else ValueName
                return {
                    "Res": {"qz_yc": [], "qz_yx": []},
                    "status": "fail",
                    "error": "没有找到标头" + str(id_label) + " " + str(value_label)
                }
            # Step 5: 识别点号列 & 数值列
            ID_labels = crop_same_col_below(ta, cell_boxes_2d, ID_coord, self.TextRecModel, True)
            value_labels = crop_same_col_below(ta, cell_boxes_2d, value_coord, self.TextRecModel, True)
            # 传入 cell_boxes_2d，让结果带上 value_box
            reslist = match_id_and_value(ID_labels, value_labels, ID_coord, value_coord, cell_boxes_2d)

            res = {"qz_yc": [], "qz_yx": []}
            if ycyx == "yc":
                res["qz_yc"] = reslist
            else:
                res["qz_yx"] = reslist

            return {
                "Res": res,
                "status": "success",
                "error": "",
                "visualization": vis_info
            }
        else:
            return {
                "Res": {"qz_yc": [], "qz_yx": []},
                "status": "fail",
                "error": "没有找到表",
                "visualization": vis_info
            }



def main():
    # ===================== 1. 配置路径（你只需要改这里）=====================
    IMAGE_PATH = "/workspace/capture_frame/capture_1.jpg"        # 测试表格图片
    SAVE_VIS_PATH = "/workspace/table_result.jpg"   # 可视化结果保存路径

    # 模型路径配置（和你实际路径一致）
    scada_model_config = {
        "TextRecModelpath": r"/workspace/scadaandqz/models/rec/PP-OCRv5_mobile_rec_infer" ,       # 文字识别模型
        "TextDetModelpath": r"/workspace/scadaandqz/models/det/PP-OCRv5_server_det_infer",       # 文字检测模型
        "tabelmodelpath": "/workspace/scadaandqz/models/tabledet.pt" ,     # 你的表格检测模型,
        "wzqymodelpath":"/workspace/scadaandqz/models/det_num_sl.pt"
    }

    # 识别参数（点号列名、值列名）
    params = {
        "idname": "数据点号",        # 你要匹配的ID列名称
        "valuenameyc": "数据值", # 遥测值列名
        "valuenameyx": "数据值", # 遥控值列名
        "ycyx": "yx"             # yc=遥测 / yx=遥控
    }

    # ===================== 2. 初始化OCR引擎 =====================
    print("🔍 初始化OCR识别器...")
    try:
        ocr = OCRRecognizer(scada_model_config)
        print("✅ OCR模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # ===================== 3. 读取图片 =====================
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"❌ 图片读取失败: {IMAGE_PATH}")
        return

    # ===================== 4. 执行识别 =====================
    print("\n🚀 开始表格识别...")
    result = ocr.run(frame, params)

    # ===================== 5. 输出结果 =====================
    print("\n" + "="*50)
    print("📊 识别结果：")
    print("状态:", result["status"])
    if result["status"] == "success":
        print("✅ 识别成功！")
        print("数据列表：")
        for item in result["Res"]["qz_yc"]:
            print(f"   ID: {item['id']:10} | Value: {item['value']}")
        for item in result["Res"]["qz_yx"]:
            print(f"   ID: {item['id']:10} | Value: {item['value']}")
    else:
        print("❌ 识别失败:", result["error"])

    print("="*50)

    # ===================== 6. 可视化表格线（可选）=====================
    try:
        h_ys, v_xs, table_img = ocr.tabelmodel.run(frame)
        if h_ys and v_xs:
            visualize_table_lines(table_img, h_ys, v_xs, SAVE_VIS_PATH)
            print(f"\n📸 表格线可视化已保存: {SAVE_VIS_PATH}")
    except:
        pass

    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()
