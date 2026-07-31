import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np
import pandas as pd
import os
import logging
import codecs
import xml.etree.ElementTree as ET
import re
import math
import json
from collections import deque
import cv2
from functools import lru_cache
from common.func import cv2_imwrite_chinese
from common.global_data_index import GlobalDataIndex
# ===================== 全局日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== 1. 高频操作缓存优化（核心提速点） =====================
# 图元文件读取缓存：同一个图元只读取1次，避免重复IO
@lru_cache(maxsize=1024)
def extract_wh_aligncenter_cached(file_path: str):
    """带缓存的图元尺寸提取，同一个文件只解析一次"""
    pattern = re.compile(
        r'w="([^"]+)"|h="([^"]+)"|AlignCenter="([^"]+)"',
        re.IGNORECASE
    )
    result = {}
    if not os.path.exists(file_path):
        return result
    try:
        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if all(key in line for key in ['w=', 'h=', 'AlignCenter=']):
                    matches = pattern.findall(line)
                    for w_val, h_val, ac_val in matches:
                        if w_val: result['w'] = w_val
                        if h_val: result['h'] = h_val
                        if ac_val: result['AlignCenter'] = ac_val
                    break
    except Exception as e:
        logger.warning(f"读取图元文件出错：{e}")
    return result

# 正则匹配缓存：同一个正则表达式只编译1次
PATTERN_CACHE = {
    "rotate_scale": re.compile(r'rotate\(\s*(-?\d+\.?\d*)\s*\)\s*scale\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'),
    "g_dimension": re.compile(r'<G\s+[^>]*?w\s*=\s*["\']([^"\']*)["\'].*?h\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE | re.DOTALL),
    "g_w": re.compile(r' w\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE),  # 提取宽度
    "g_h": re.compile(r' h\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE),  # 提取高度
    "g_tag": re.compile(r'<G\s[^>]+>', re.IGNORECASE | re.DOTALL),      # 定位 G 标签
    # "g_dimension": re.compile(
    #     r'<G\s[^>]*?w\s*=\s*["\']([^"\']*)["\'][^>]*?h\s*=\s*["\']([^"\']*)["\']',
    #     re.IGNORECASE | re.DOTALL
    # ),
    "target_values": re.compile(
        r'col=(\d+).*?ln=(\d+).*?bay_id=(\d+)L?.*?dx=(\d+)',
        re.IGNORECASE | re.DOTALL
    )
}

# ===================== 2. 坐标/图像处理工具函数（无修改，仅保留核心） =====================
def parse_coordinate_string(coord_str):
    """提取G文件坐标信息"""
    if not coord_str:
        return None
    try:
        clean_str = coord_str.replace("，", ",").strip()
        coord_parts = [part for part in clean_str.split() if part.strip()]
        nums = []
        for part in coord_parts:
            xy = [x.strip() for x in part.split(",") if x.strip()]
            if len(xy) != 2:
                raise ValueError(f"坐标段{part}格式错误")
            x = int(float(xy[0]))   # 先转 float，再转 int
            y = int(float(xy[1]))
            nums.append(x)
            nums.append(y)
        return nums
    except Exception as e:
        logger.warning(f"解析坐标失败：{e}（输入：{coord_str}）")
        return None

def process_low_gray_regions(image):
    """提取画面最大黑色区域"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    close = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxrect = ()
    maxarea = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > maxarea:
            maxarea = area
            maxrect = (x, y, w, h)
    return maxrect

def get_scale_and_translate_matrix(original_xywh, target_xywh):
    ox, oy, ow, oh = original_xywh
    original_cx = ox + ow / 2
    original_cy = oy + oh / 2
    tx, ty, tw, th = target_xywh
    target_cx = tx + tw / 2
    target_cy = ty + th / 2
    scale_w = tw / ow
    scale_h = th / oh
    scale = min(scale_w, scale_h)
    dx = target_cx - original_cx
    dy = target_cy - original_cy
    M_scale = cv2.getRotationMatrix2D((original_cx, original_cy), 0, scale)
    M_scale[0, 2] += dx
    M_scale[1, 2] += dy
    return M_scale, scale

def jx_M(image, size):
    """计算M变化矩阵"""
    gw, gh = size
    results = process_low_gray_regions(image)
    # results = (0,83,1920,933)
    if not results:
        return np.eye(2, 3, dtype=np.float32)
    M, scale = get_scale_and_translate_matrix([0, 0, int(gw), int(gh)], results)
    return M

def M_jsuan(rect, M):
    """G文件坐标映射到实际画面坐标"""
    try:
        if len(rect) % 2 != 0:
            return []
        points = np.array([
            [rect[i], rect[i+1]] for i in range(0, len(rect), 2)
        ], dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.transform(points, M)
        transformed_flat = transformed_points.reshape(-1).tolist()
        return [int(round(coord)) for coord in transformed_flat]
    except Exception as e:
        logger.warning(f"坐标变换失败：{e}")
        return []

def rotate_rectangle(cx, cy, w, h, scale_x, scale_y, angle_deg):
    """
    计算旋转后的外接矩形
    1、将矩形四个角点相对中心点做缩放
    2、绕中心点旋转 angle_deg 度
    3、计算旋转后四个点的最小外接矩形
    """
    angle_rad = math.radians(angle_deg)
    cos_ang = math.cos(angle_rad)
    sin_ang = math.sin(angle_rad)
    scaled_half_w = (w / 2) * scale_x
    scaled_half_h = (h / 2) * scale_y
    points = [
        (cx - scaled_half_w, cy - scaled_half_h),
        (cx + scaled_half_w, cy - scaled_half_h),
        (cx + scaled_half_w, cy + scaled_half_h),
        (cx - scaled_half_w, cy + scaled_half_h)
    ]
    rotated_points = []
    for (x, y) in points:
        tx = x - cx
        ty = y - cy
        rx = tx * cos_ang - ty * sin_ang
        ry = tx * sin_ang + ty * cos_ang
        rotated_x = rx + cx
        rotated_y = ry + cy
        rotated_points.append((rotated_x, rotated_y))
    rotated_xs = [p[0] for p in rotated_points]
    rotated_ys = [p[1] for p in rotated_points]
    new_x1 = min(rotated_xs)
    new_y1 = min(rotated_ys)
    new_w = max(rotated_xs) - new_x1
    new_h = max(rotated_ys) - new_y1
    return (new_x1, new_y1, new_w, new_h)

def jisuan(x, y, w, h, rotate, scale_x, scale_y):
    cx = x + (w/2)
    cy = y + (h/2)
    return rotate_rectangle(cx, cy, w, h, scale_x, scale_y, rotate)

def calculate_string_size(text, char_size=18):
    """计算给定文本的宽高"""
    lines = text.split('\n')
    line_widths = []
    digit_width = char_size * 0.65
    for line in lines:
        line_pixel_width = 0
        for char in line:
            line_pixel_width += digit_width if char.isdigit() else char_size
        line_widths.append(line_pixel_width)
    total_width = max(line_widths) if line_widths else 0
    total_height = len(lines) * char_size
    return (total_width, total_height), {}

def cv2_imread_chinese(path, flags=cv2.IMREAD_COLOR):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, flags)
        return img
    except Exception as e:
        logger.warning(f"读取中文路径图像出错：{e}")
        return None

def calculate_rect_iou(rect1, rect2) -> float:
    if len(rect1) != 4 or len(rect2) != 4:
        return 0.0
    try:
        x1_1, y1_1, x2_1, y2_1 = map(float, rect1)
        x1_2, y1_2, x2_2, y2_2 = map(float, rect2)
    except ValueError:
        return 0.0
    if x2_1 <= x1_1 or y2_1 <= y1_1 or x2_2 <= x1_2 or y2_2 <= y1_2:
        return 0.0
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1
    inter_area = inter_width * inter_height if inter_width > 0 and inter_height > 0 else 0.0
    rect1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    rect2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = rect1_area + rect2_area - inter_area
    return round(inter_area / union_area, 6) if union_area != 0 else 0.0



# ===================== 4. 你的原版jxp_NameString函数（适配全局索引版） =====================
def parse_instruction(instruction: str) -> dict:
    """解析原始指令，提取col、ln、sql"""
    result = {"col": None, "ln": None, "sql": None, "dx": None, "yx": []}
    param_items = instruction.split(";")
    for item in param_items:
        item = item.strip()
        if not item:
            continue
        key_value = item.split("=", 1)
        if len(key_value) < 2:
            continue
        key = key_value[0].strip().lower()
        value = key_value[1].strip()
        if key == "col":
            try:
                result["col"] = int(value)
            except ValueError:
                print(f"警告：col值'{value}'非有效数字")
        elif key == "ln":
            try:
                result["ln"] = int(value)
            except ValueError:
                print(f"警告：ln值'{value}'非有效数字")
        elif key == "sql":
            result["sql"] = value
        elif key == "dx":
            try:
                result["dx"] = int(value)
            except:
                result["dx"] = value
    return result

def parse_sql(sql_str: str) -> dict:
    """解析SQL，提取表名、bay_id、name_like"""
    sql_lower = sql_str.lower()
    result = {"table_name": None, "bay_id": None, "name_like": None}
    from_match = re.search(r"from\s+(\w+)", sql_lower)
    if from_match:
        result["table_name"] = from_match.group(1)
    bay_id_match = re.search(r"bay_id\s*=\s*(\d+)(?:L|l)?", sql_str)
    if bay_id_match:
        result["bay_id"] = bay_id_match.group(1)
    name_like_match = re.search(r"name\s+like\s*['\"]%([^%]+)%['\"]", sql_str, re.IGNORECASE)
    if name_like_match:
        result["name_like"] = name_like_match.group(1)
    return result

def jxp_NameString(input_instruction: str, global_index: GlobalDataIndex):
    """
    解析G文化SQL语句，在RelaySignal中查找原始信号，再通过Discrete映射到具体点号
    你的原版逻辑，适配全局索引，零文件IO
    """
    # 1. 解析指令
    instruction_data = parse_instruction(input_instruction)
    if not instruction_data["sql"]:
        return instruction_data
    
    # 2. 解析SQL
    sql_data = parse_sql(instruction_data["sql"])
    target_bay_id = sql_data["bay_id"]
    target_name = sql_data["name_like"]
    table_name = sql_data["table_name"]
    
    if not target_bay_id or global_index.relay_df is None:
        return instruction_data

    print(f"=== 解析到的关键参数 ===")
    print(f"目标bay_id：{target_bay_id}")
    print(f"pathName需包含：{target_name}")

    # 3. 在预加载的RelaySignal表中筛选（零IO）
    df = global_index.relay_df.copy()
    
    # 找到BAYID列
    bay_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if "bayid" in col_lower or "bay" in col_lower:
            bay_col = col
            break
    if bay_col is None:
        print("⚠️ 未找到BAYID列")
        return instruction_data

    # 找到mRID列
    mrid_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if "mrid" in col_lower:
            mrid_col = col
            break
    if mrid_col is None:
        print("⚠️ 未找到mRID列")
        return instruction_data

    # 构建筛选条件
    df["bay_str"] = df[bay_col].astype(str)
    if not target_name:
        filter_condition = (df["bay_str"] == target_bay_id)
    else:
        # 找到pathName列
        pathname_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if "pathname" in col_lower:
                pathname_col = col
                break
        if pathname_col is None:
            print("⚠️ 未找到pathName列")
            return instruction_data
        filter_condition = (
            (df["bay_str"] == target_bay_id) &
            (df[pathname_col].str.contains(target_name, na=False))
        )

    # 4. 提取第一轮mRID
    filtered_df = df[filter_condition]
    if filtered_df.empty:
        print("ℹ️ 未找到符合条件的记录")
        return instruction_data
    
    mrid_list_1 = filtered_df[mrid_col].dropna().tolist()
    print(f"✅ 第一轮找到 {len(mrid_list_1)} 个mRID")

    # 5. 在Discrete表中匹配devID（使用预构建的映射，O(1)）
    mrid_list_2 = []
    for original_mrid in mrid_list_1:
        original_mrid_clean = str(original_mrid).strip()
        if original_mrid_clean in global_index.discrete_devid_mrid_map:
            dev_mrid = global_index.discrete_devid_mrid_map[original_mrid_clean]
            mrid_list_2.append(dev_mrid)
            print(f"✅ 原表mRID[{mrid_list_1.index(original_mrid)}] {original_mrid} → dev表格mRID：{dev_mrid}")
        else:
            print(f"ℹ️ 原表mRID[{mrid_list_1.index(original_mrid)}] {original_mrid} → 未匹配到devID表格")

    instruction_data["yx"] = mrid_list_2
    return instruction_data
# ===================== 5. 重构后的G文件解析类（纯内存计算，零IO） =====================
class AnalyseG2:
    """
    重构后：仅接收预加载的全局索引，不做任何文件IO，全程纯内存计算
    彻底消灭递归内的重复初始化和IO开销
    """
    def __init__(self, g_path: str, global_index: GlobalDataIndex, button_config: dict = None):
        self.g_path = g_path
        self.global_index = global_index  # 全局索引，全程复用
        self.button_config = button_config or {}
        self.bottun_g = list(self.button_config.keys())
        self.bottun_needyx = list(self.button_config.values())
        self.points = []
        self.root = None
        self.gw = 0
        self.gh = 0
        self.size = [0, 0]
        # 图元偏移配置
        self.g_icn = {
        }

        # 解析XML（仅一次IO）
        self._parse_xml()
        # 提取G画布尺寸
        self._extract_g_dimensions()

    def _parse_xml(self):
        """仅一次XML文件读取"""
        try:
            with codecs.open(self.g_path, 'r', encoding='gbk', errors='ignore') as file:
                xml_content = file.read()
            self.root = ET.fromstring(xml_content) if xml_content else None
        except Exception as e:
            logger.warning(f"解析XML文件失败: {e}，路径：{self.g_path}")
            self.root = None

    def _extract_g_dimensions(self):
        """自动提取G画布宽高，w=宽度 h=高度，顺序无关"""
        if not os.path.exists(self.g_path):
            return
        try:
            with open(self.g_path, 'r', encoding='GBK', errors='ignore') as f:
                content = f.read()

            # 1. 先找到 <G ...> 标签
            print(self.g_path)
            g_tag = PATTERN_CACHE["g_tag"].search(content)
            if not g_tag:
                logger.warning("未找到G标签")
                return

            # 2. 自动提取 w 和 h（不管顺序！）
            g_content = g_tag.group()
            w_match = PATTERN_CACHE["g_w"].search(g_content)
            h_match = PATTERN_CACHE["g_h"].search(g_content)

            # 3. 自动赋值：w=宽度，h=高度
            if w_match and h_match:
                self.gw = int(float(w_match.group(1).strip()))  # 自动拿宽度
                self.gh = int(float(h_match.group(1).strip()))  # 自动拿高度
                self.size = [self.gw, self.gh]
                logger.info(f"自动提取成功：宽={self.gw}, 高={self.gh}")

        except Exception as e:
            logger.warning(f"提取G尺寸失败：{e}")

    def matech(self, tfr):
        """解析旋转缩放，使用预编译正则"""
        if not tfr:
            return 0, 1.0, 1.0
        match = PATTERN_CACHE["rotate_scale"].search(tfr)
        if match:
            try:
                rotate_num = int(match.group(1))
                scale_w = abs(float(match.group(2)))
                scale_h = abs(float(match.group(3)))
                return rotate_num, scale_w, scale_h
            except (ValueError, TypeError):
                pass
        return 0, 1.0, 1.0

    def get_normal_point_pname(self, label, w=40, h=20, mode="yx"):
        """带p_NameString解析的元素提取，使用全局O(1)索引"""
        if not self.root:
            return
        dtexts = self.root.findall(f'.//{label}')
        for dtext in dtexts:
            tyname = ""
            devref = dtext.get('devref', '')
            tfr = dtext.get('tfr', '')
            keyid = dtext.get('keyid', '')
            rtkeyid=dtext.get('rtkeyid', '')
            name = dtext.get('key_name', '')
            p_NameString = dtext.get('p_NameString', '')
            gid = dtext.get('id', '')
            rect = [0, 0, w, h]

            # 解析p_NameString，从全局索引获取keyid
            jxpdict = jxp_NameString(p_NameString, self.global_index)
            # print(jxpdict)
            if jxpdict["yx"]:
                keyid = jxpdict["yx"][0]

            # 解析坐标
            if devref == "":
                # 无图元引用
                dtext_x1 = float(dtext.get('x1', 0.0))
                dtext_y1 = float(dtext.get('y1', 0.0))
                dtext_x2 = float(dtext.get('x2', 0.0))
                dtext_y2 = float(dtext.get('y2', 0.0))
                if dtext_x1 > 1 or dtext_y1 > 1 or dtext_x2 > 1 or dtext_y2 > 1:
                    rect = [int(dtext_x1), int(dtext_y1), int(dtext_x2), int(dtext_y2)]
                else:
                    d = dtext.get('d', '')
                    if d:
                        xys = parse_coordinate_string(d)
                        if xys and len(xys) >= 4:
                            rect = xys
                    else:
                        dtext_x = float(dtext.get('x', 0.0))
                        dtext_y = float(dtext.get('y', 0.0))
                        w = int(dtext.get('w', w))
                        h = int(dtext.get('h', h))
                        rect = [int(dtext_x), int(dtext_y), int(dtext_x + w), int(dtext_y + h)]
            else:
                # 引用图元
                dtext_x = float(dtext.get('x', 0.0))
                dtext_y = float(dtext.get('y', 0.0))
                rotate_num, scale_w, scale_h = self.matech(tfr)
                ty_message = {}
                dx, dy, dw, dh = [0, 0, 0, 0]

                if devref:
                    tyname = devref.split(':')[0].split('#')[-1] if ':' in devref and '#' in devref else ''
                    typath = self.global_index.find_file(tyname)
                    if typath:
                        ty_message = extract_wh_aligncenter_cached(typath)
                    if tyname in self.g_icn:
                        dx, dy, dw, dh = self.g_icn[tyname]

                w = int(ty_message.get('w', w)) + dw
                h = int(ty_message.get('h', h)) + dh
                dtext_x += dx
                dtext_y += dy
                if "测试信号光子牌.gzp.icn.g" in devref:
                    dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x+2, dtext_y+2, 42, 42, rotate_num, scale_w, scale_h)
                elif "SN_引线柜手车.gld.icn.g" in devref:
                    dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x-1, dtext_y+4, 14, 20, rotate_num, scale_w, scale_h)
                elif "SN_接地刀闸1.jdd.icn.g" in devref:
                    dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x, dtext_y-1, 52, 22, rotate_num, scale_w, scale_h)
                elif "SN_刀闸_竖2.gld.icn.g" in devref:
                    dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x-2, dtext_y+1, 18, 42, rotate_num, scale_w, scale_h)
                else:
                    dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x-2, dtext_y, w+5, h+4, rotate_num, scale_w, scale_h)
                rect = [int(dtext_x), int(dtext_y), int(dtext_x + dtext_w), int(dtext_y + dtext_h)]

            # 按钮判断
            have_button = any(btn in devref for btn in self.bottun_g)
            need_ycyx = False
            if have_button:
                for i, btn in enumerate(self.bottun_g):
                    if btn in devref and i < len(self.bottun_needyx):
                        need_ycyx = self.bottun_needyx[i]
                        break
                if not need_ycyx:
                    mode = "draw"

            # 构建结果
            tree = {
                "cimeid": [], "cimename": [], "cimeboxs": [], "cimetypes": [],
                "name": name, "gid": gid, "box": rect, "ty": devref, "lx": label,
                "ycyx": "draw", "tyname": tyname, "isbutton": have_button
            }

            # 遥信匹配（O(1)，零循环）
            if mode == "yx" and (keyid or rtkeyid):
                if(rtkeyid!=""):
                    yx_info = self.global_index.yx_keyid_map.get(rtkeyid)
                else:
                    yx_info = self.global_index.yx_keyid_map.get(keyid)
                if yx_info:
                    cimetype = "dollybreaker_sc" if "SN_引线柜手车.gld.icn.g" in devref else label
                    tree["cimeid"] = [yx_info["点号"]]
                    tree["cimename"] = [yx_info["名称"]]
                    tree["间隔名"] = [yx_info["间隔名"]]
                    tree["站名"] = [yx_info["站名"]]
                    tree["cimeboxs"] = [rect]
                    tree["cimetypes"] = [cimetype.lower()]
                    tree["ycyx"] = "yx"
            self.points.append(tree)

    def get_normal_point(self, label, w=40, h=20, mode="draw"):
        """普通元素提取，无IO纯内存计算"""
        if not self.root:
            return
        dtexts = self.root.findall(f'.//{label}')

        for dtext in dtexts:
            tyname = ""
            devref = dtext.get('devref', '')
            tfr = dtext.get('tfr', '')
            name = dtext.get('key_name', '')
            gid = dtext.get('id', '')
            rect = [0, 0, w, h]

            if devref == "":
                dtext_x1 = float(dtext.get('x1', 0.0))
                dtext_y1 = float(dtext.get('y1', 0.0))
                dtext_x2 = float(dtext.get('x2', 0.0))
                dtext_y2 = float(dtext.get('y2', 0.0))
                if dtext_x1 > 1 or dtext_y1 > 1 or dtext_x2 > 1 or dtext_y2 > 1:
                    rect = [int(dtext_x1), int(dtext_y1), int(dtext_x2), int(dtext_y2)]
                else:
                    d = dtext.get('d', '')
                    if d:
                        xys = parse_coordinate_string(d)
                        if xys and len(xys) >= 4:
                            rect = xys
                    else:
                        dtext_x = float(dtext.get('x', 0.0))
                        dtext_y = float(dtext.get('y', 0.0))
                        w = int(float(dtext.get('w', w)))
                        h = int(float(dtext.get('h', h)))
                        rect = [int(dtext_x), int(dtext_y), int(dtext_x + w), int(dtext_y + h)]
            else:
                dtext_x = float(dtext.get('x', 0.0))
                dtext_y = float(dtext.get('y', 0.0))
                rotate_num, scale_w, scale_h = self.matech(tfr)
                ty_message = {}
                dx, dy, dw, dh = [0, 0, 0, 0]

                if "SN_引线柜手车.gld.icn.g" in devref:
                    tyname = devref.split(':')[0].split('#')[-1] if ':' in devref and '#' in devref else ''
                    typath = self.global_index.find_file(tyname)
                    if typath:
                        ty_message = extract_wh_aligncenter_cached(typath)
                    if tyname in self.g_icn:
                        dx, dy, dw, dh = self.g_icn[tyname]

                w = int(ty_message.get('w', w)) + dw
                h = int(ty_message.get('h', h)) + dh
                dtext_x += dx
                dtext_y += dy
                dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x, dtext_y, w, h, rotate_num, scale_w, scale_h)
                rect = [int(dtext_x), int(dtext_y), int(dtext_x + dtext_w), int(dtext_y + dtext_h)]

            have_button = any(btn in devref for btn in self.bottun_g)
            tree = {
                "cimeid": [], "cimename": [], "cimeboxs": [], "cimetypes": [],
                "name": name, "gid": gid, "box": rect, "ty": devref, "lx": label,
                "ycyx": "draw", "tyname": tyname, "isbutton": have_button
            }
            self.points.append(tree)

    def get_Text(self):
        """文本元素提取"""
        if not self.root:
            return
        dtexts = self.root.findall('.//Text')
        for dtext in dtexts:
            name = dtext.get('ts', '')
            gid = dtext.get("id", '')
            wm=dtext.get('wm')
            p_FontWidth = int(float(dtext.get('p_FontWidth', 12)))
            dtext_x = float(dtext.get('x', 0.0))
            dtext_y = float(dtext.get('y', 0.0))
            (total_width, total_height), _ = calculate_string_size(name, p_FontWidth)
            if wm=="1":
                dtext_h = total_height
                dtext_w = total_width
            else:
                dtext_w = total_height
                dtext_h = total_width
            rect = (int(dtext_x), int(dtext_y), int(dtext_x + dtext_w), int(dtext_y + dtext_h))
            tree = {
                "cimeid": [], "cimename": [], "cimeboxs": [], "cimetypes": [],
                "name": name, "gid": gid, "box": rect, "ty": '', "lx": "Text",
                "ycyx": "draw"
            }
            self.points.append(tree)

    def get_DText(self):
        """遥测文本提取，O(1)匹配"""
        if not self.root:
            return
        dtexts = self.root.findall('.//DText')
        for dtext in dtexts:
            dotlength = int(dtext.get("dotlength", 0)) + 3.5
            dtext_x = float(dtext.get('x', 0.0))
            dtext_y = float(dtext.get('y', 0.0))
            rtkeyid = dtext.get('rtkeyid', '')
            
            fs = int(dtext.get('fs', 12))
            dtext_h = fs
            dtext_w = int(dtext_h * dotlength * 0.72)
            name = dtext.get('key_name', '')
            gid = dtext.get('id', '')
            keyid = dtext.get('keyid', '')
            rect = (int(dtext_x)-8, int(dtext_y), int(dtext_x + dtext_w), int(dtext_y + dtext_h+3))
            tree = {
                "cimeid": [], "cimename": [], "cimeboxs": [], "cimetypes": [],
                "name": name, "gid": gid, "box": rect, "ty": "", "lx": "DText",
                "ycyx": "yx"
            }
            # 遥测匹配（O(1)）
            if keyid or rtkeyid:
                if(rtkeyid!=""):
                    yc_info = self.global_index.yc_keyid_map.get(rtkeyid)
                else:
                    yc_info = self.global_index.yc_keyid_map.get(keyid)
                if yc_info:
                    tree["cimeid"] = [yc_info["点号"]]
                    tree["cimename"] = [yc_info["名称"]]
                    tree["间隔名"] = [yc_info["间隔名"]]
                    tree["站名"] = [yc_info["站名"]]
                    tree["cimeboxs"] = [rect]
                    tree["cimetypes"] = ["dtext"]
                    tree["ycyx"] = "yc"
            self.points.append(tree)

    def get_DollyBreaker(self):
        """小车开关提取，O(1)匹配"""
        if not self.root:
            return
        dtexts = self.root.findall('.//DollyBreaker')
        for dtext in dtexts:
            devref = dtext.get('devref', '')
            tfr = dtext.get('tfr', '')
            gid = dtext.get('id', '')
            name1 = dtext.get('key_name1', '')
            name2 = dtext.get('key_name2', '')
            keyid1 = dtext.get('keyid1', '')
            keyid2 = dtext.get('keyid2', '')
            rtkeyid1 = dtext.get('rtkeyid1', '')
            rtkeyid2 = dtext.get('rtkeyid2', '')
            dtext_x = float(dtext.get('x', 0.0))
            dtext_y = float(dtext.get('y', 0.0))
            rotate_num, scale_w, scale_h = self.matech(tfr)

            w, h = 40, 20
            tyname = ""
            if devref:
                tyname = devref.split(':')[0].split('#')[-1] if ':' in devref and '#' in devref else ''
                typath = self.global_index.find_file(tyname)
                if typath:
                    ty_message = extract_wh_aligncenter_cached(typath)
                    w = int(ty_message.get('w', w))
                    h = int(ty_message.get('h', h))

            if tyname == "SH_手车开关sh.xck.icn.g":
                dtext_x1, dtext_y1, dtext_w1, dtext_h1 = jisuan(dtext_x, dtext_y+19, w, 22, rotate_num, scale_w, scale_h)
                dtext_x2, dtext_y2, dtext_w2, dtext_h2 = jisuan(dtext_x, dtext_y-5, w, 12, rotate_num, scale_w, scale_h)
            elif tyname=="SH_CT_小车开关.xck.icn.g":
                dtext_x1, dtext_y1, dtext_w1, dtext_h1 = jisuan(dtext_x, dtext_y+28, w, 22, rotate_num, scale_w, scale_h)
                dtext_x2, dtext_y2, dtext_w2, dtext_h2 = jisuan(dtext_x, dtext_y, w, 22, rotate_num, scale_w, scale_h)
            else:
                dtext_x1, dtext_y1, dtext_w1, dtext_h1 = jisuan(dtext_x, dtext_y+19, w, 22, rotate_num, scale_w, scale_h)
                dtext_x2, dtext_y2, dtext_w2, dtext_h2 = jisuan(dtext_x, dtext_y, w, 18, rotate_num, scale_w, scale_h)
            dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x, dtext_y, w, h, rotate_num, scale_w, scale_h)

            rect = (int(dtext_x), int(dtext_y), int(dtext_x+dtext_w), int(dtext_y+dtext_h))
            rect1 = (int(dtext_x1), int(dtext_y1), int(dtext_x1+dtext_w1), int(dtext_y1+dtext_h1))
            rect2 = (int(dtext_x2), int(dtext_y2), int(dtext_x2+dtext_w2), int(dtext_y2+dtext_h2))

            have_button = any(btn in devref for btn in self.bottun_g)
            tree = {
                "cimeid": [], "cimename": [], "cimeboxs": [], "cimetypes": [],
                "name": name1, "gid": gid, "box": rect, "ty": devref, "lx": "DollyBreaker",
                "ycyx": "yx", "tyname": tyname, "isbutton": have_button
            }

            # 遥信匹配（O(1)）
            if gid and name1 and name2:
                if(rtkeyid1!=""):
                    yx_info1 = self.global_index.yx_keyid_map.get(rtkeyid1)
                    yx_info2 = self.global_index.yx_keyid_map.get(rtkeyid2)
                else:
                    yx_info1 = self.global_index.yx_keyid_map.get(keyid1)
                    yx_info2 = self.global_index.yx_keyid_map.get(keyid2)
                if yx_info1 and yx_info2:
                    tree["cimeid"] = [yx_info1["点号"], yx_info2["点号"]]
                    tree["cimename"] = [yx_info1["名称"], yx_info2["名称"]]
                    tree["间隔名"] = [yx_info1["间隔名"],yx_info2["间隔名"]]
                    tree["站名"] = [yx_info1["站名"],yx_info2["站名"]]
                    tree["cimeboxs"] = [rect1, rect2]
                    tree["cimetypes"] = ["dollybreaker_kg", "dollybreaker_sc"]
            self.points.append(tree)

    def get_poke(self):
        """跳转按钮提取"""
        if not self.root:
            return
        dtexts = self.root.findall('.//poke')
        for dtext in dtexts:
            dtext_x = float(dtext.get('x', 0.0))
            dtext_y = float(dtext.get('y', 0.0))
            dtext_h = float(dtext.get('h', 0))
            dtext_w = float(dtext.get('w', 0))
            ahref = dtext.get('ahref', 'back')
            gid = dtext.get("id", '')
            if abs(dtext_h) < 1 or abs(dtext_w) < 1:
                continue
            rect = [min(dtext_x,dtext_x + dtext_w), min(dtext_y,dtext_y + dtext_h), max(dtext_x,dtext_x + dtext_w), max(dtext_y,dtext_y + dtext_h)]
            maxiou = 0.1
            newname = ""
            if not ahref:
                for point in self.points:
                    if point["lx"] == 'Text' and point["name"]:
                        textrect = point["box"]
                        iou = calculate_rect_iou(textrect, rect)
                        if iou > maxiou:
                            newname = point["name"]
                            maxiou = iou
                ahref = newname if newname else ahref
            if not ahref:
                continue
            tree = {
                "cimeid": [], "cimename": [], "cimeboxs": [], "cimetypes": [],
                "name": ahref, "gid": gid, "box": rect, "ty": "", "lx": "poke",
                "ycyx": "draw", "isbutton": True
            }
            self.points.append(tree)

    def getSpecialGZP(self):
        """光字牌分页处理，使用全局索引，零IO"""
        if not self.root:
            return []
        dtexts = self.root.findall(f'.//Gzp')
        page_Points = []
        for dtext in dtexts:
            gid = dtext.get("id", '')
            devref = dtext.get('devref', '')
            tfr = dtext.get('tfr', '')
            p_NameString = dtext.get('p_NameString', '')
            if not p_NameString:
                continue
            dtext_x = float(dtext.get('x', 0.0))
            dtext_y = float(dtext.get('y', 0.0))
            rotate_num, scale_w, scale_h = self.matech(tfr)

            tyname = ""
            w, h = 10, 10
            if devref:
                tyname = devref.split(':')[0].split('#')[-1] if ':' in devref and '#' in devref else ''
                typath = self.global_index.find_file(tyname)
                if typath:
                    ty_message = extract_wh_aligncenter_cached(typath)
                    w = int(ty_message.get('w', w))
                    h = int(ty_message.get('h', h))
            if "测试信号光子牌.gzp.icn.g" in devref:
                dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x, dtext_y-2, 40, 40, rotate_num, scale_w, scale_h)
            else:
                dtext_x, dtext_y, dtext_w, dtext_h = jisuan(dtext_x, dtext_y, w, h, rotate_num, scale_w, scale_h)

            # 从全局索引获取数据
            jxpdict = jxp_NameString(p_NameString, self.global_index)
            ln_val = jxpdict.get('ln', 1)
            if ln_val is None:
                ln_val = 20
            col_val = jxpdict.get('col', 1)
            dx_val = jxpdict.get('dx', 548)
            if(dx_val is None):
                dx_val=548
            relay_mrid = jxpdict.get('yx', [])

            if not relay_mrid:
                continue
            max_capacity = ln_val * col_val
            total_pages = (len(relay_mrid) + max_capacity - 1) // max_capacity
            for page in range(1, total_pages + 1):
                one_page = []
                start_idx = (page - 1) * max_capacity
                end_idx = min(page * max_capacity, len(relay_mrid))
                for idx_in_page in range(end_idx - start_idx):
                    global_idx = start_idx + idx_in_page
                    dtext_x1 = dtext_x + dx_val * (idx_in_page // ln_val)
                    dtext_y1 = dtext_y + (dtext_h+25) * (idx_in_page % ln_val)
                    discrete_mrid = relay_mrid[global_idx]
                    if not discrete_mrid:
                        continue
                    # O(1)匹配遥信点
                    yx_info = self.global_index.yx_keyid_map.get(discrete_mrid)
                    rect = [dtext_x1, dtext_y1, dtext_x1 + dtext_w, dtext_y1 + dtext_h]
                    if yx_info:
                        tree = {
                            "cimeid": yx_info["点号"],
                            "cimename": yx_info["名称"],
                            "cimeboxs": rect,
                            "cimetypes": "gzp",
                            "ycyx": "yx",
                            "tyname": tyname
                        }
                        one_page.append(tree)
                if one_page:
                    page_Points.append(one_page)
        return page_Points

    def getallpoint(self):
        """统一提取所有元素"""
        if not self.root:
            logger.warning(f"XML根节点为空，跳过点提取：{self.g_path}")
            return
        # 遥信类元素
        yx_labels = ["Protect", "GroundDisconnector", "Disconnector", "CBreaker", "Gzp", "Terminal"]
        for label in yx_labels:
            self.get_normal_point_pname(label, mode="yx")
        # 绘制类元素
        draw_labels = [
            "Bus", "ConnectLine", "Transformer2", "Capacitor_P","rect","line","Transformer3",
            "Reactor_P", "Arrester", "PT", "ACLineEnd", "EnergyConsumer", "image","Fuse"
        ]
        for label in draw_labels:
            self.get_normal_point(label, mode="draw")
        # 文本类元素
        self.get_Text()
        self.get_DText()
        # 特殊元素
        self.get_DollyBreaker()
        self.get_poke()


    def getddpoints(self, yx_labels=["Protect", "GroundDisconnector", "Disconnector", "DollyBreaker", "CBreaker", "Gzp"], yc_labels=["DText"]):
        """提取遥信遥测点，去重"""
        ddpoints = []
        button_dict = {}
        for point in self.points:
            if point["lx"] in yx_labels:
                for i in range(len(point["cimeid"])):
                    newpoint = {
                        "cimeid": point["cimeid"][i],
                        "cimename": point["cimename"][i],
                        "cimeboxs": point["cimeboxs"][i],
                        "cimetypes": point["cimetypes"][i],
                        "tyname": point.get('tyname', ''),
                        "ycyx": "yx",
                    }
                    ddpoints.append(newpoint)
            elif point["lx"] in yc_labels:
                for i in range(len(point["cimeid"])):
                    newpoint = {
                        "cimeid": point["cimeid"][i],
                        "cimename": point["cimename"][i],
                        "cimeboxs": point["cimeboxs"][i],
                        "cimetypes": point["cimetypes"][i],
                        "tyname": point.get('tyname', ''),
                        "ycyx": "yc",
                    }
                    ddpoints.append(newpoint)
            if point.get('isbutton', False)  and point["box"]:
                button_dict[point["name"]] = point["box"]
        return ddpoints, button_dict

    def create_button(self, btn_type, box):
        return {"type": btn_type, "box": box, "size": self.size}

    def create_node_dict(self, node_name, points=None, children_names=None, buttons=None):
        return {
            "name": node_name,
            "size": self.size,
            "children_names": children_names or [],
            "buttons": buttons or [],
            "points": points or []
        }

    def CreatMap(self, node_name):
        """创建普通节点"""
        ddpoints, button_dict = self.getddpoints()
        children_names = list(button_dict.keys())
        buttons = [self.create_button("单击", box) for box in button_dict.values()]
        return self.create_node_dict(node_name=node_name, children_names=children_names, buttons=buttons, points=ddpoints)

    def has_gzp_navigation(self):
        """判断当前 G 文件是否为光字牌总图（根据文件名）"""
        return "光字牌总图" in os.path.basename(self.g_path)

    def get_gzp_navigation_buttons(self):
        """从光字牌总图的 Gzp 图元中提取导航按钮（key_name -> 子图文件 -> 点击区域）"""
        nav_buttons = {}
        if not self.root:
            return nav_buttons
        for point in self.points:
            if point.get("lx") != "Gzp":
                continue
            child_name = point.get("name", "")
            if not child_name:
                continue
            child_path = self.global_index.find_file(child_name)
            if not child_path:
                continue
            # 只把画面文件（.pic.g，含 .bay.pic.g / .fac.pic.g）当作导航目标，排除图元 .icn.g
            if ".pic.g" not in child_path.lower():
                continue
            child_base = os.path.basename(child_path)
            box = point.get("box")
            if box and len(box) == 4:
                nav_buttons[child_base] = box
        return nav_buttons

    def CreatGZPMap(self, node_name):
        """创建光字牌总图节点：保留原有绘制/遥信点，同时把 Gzp 图元作为子节点导航按钮"""
        ddpoints, button_dict = self.getddpoints()
        gzp_nav = self.get_gzp_navigation_buttons()
        merged_buttons = {**button_dict, **gzp_nav}
        children_names = list(merged_buttons.keys())
        buttons = [self.create_button("单击", box) for box in merged_buttons.values()]
        return self.create_node_dict(
            node_name=node_name,
            children_names=children_names,
            buttons=buttons,
            points=ddpoints
        )

    def CreatJGMapbay(self, node_name, parent_name):
        """创建间隔光字牌分页节点"""
        page_Points = self.getSpecialGZP()
        #print(f'page_points:{page_Points}')
        ddpoints, button_dict = self.getddpoints(yx_labels=["Protect", "GroundDisconnector", "Disconnector", "CBreaker"])
        button_names = list(button_dict.keys())

        returnpage = None
        downpage = None
        uppage = None
        nodes = []

        if len(button_names) == 1:
            returnpage = button_dict[button_names[0]]
            if not returnpage:
                logger.warning("返回按钮不完整")
                return []
            children_names = ["BACK"]
            buttons = [self.create_button("单击", returnpage)]
            if page_Points:
                node = self.create_node_dict(node_name=node_name, children_names=children_names, buttons=buttons, points=page_Points[0] + ddpoints)
            else:
                node = self.create_node_dict(node_name=node_name, children_names=children_names, buttons=buttons, points=ddpoints)
            nodes.append(node)
            return nodes

        if len(button_names) == 3:
            for button_name in button_names:
                if button_name in ["上页", "上一页"]:
                    uppage = button_dict[button_name]
                elif button_name in ["下页", "下一页"]:
                    downpage = button_dict[button_name]
                else:
                    returnpage = button_dict[button_name]
        if not downpage or not uppage or not returnpage:
            logger.warning("分页按钮不完整")

        for i in range(len(page_Points)):
            children_names = [f"{node_name}{i+1}", parent_name]
            buttons = [self.create_button("单击", downpage), self.create_button("单击", returnpage)]
            new_nodename = node_name
            if i == 1:
                children_names.append(node_name)
                buttons.append(self.create_button("单击", uppage))
                new_nodename = f"{node_name}{i}"
            if i > 1:
                children_names.append(f"{node_name}{i-1}")
                buttons.append(self.create_button("单击", uppage))
                new_nodename = f"{node_name}{i}"
            node = self.create_node_dict(
                node_name=new_nodename,
                children_names=children_names,
                buttons=buttons,
                points=page_Points[i] + ddpoints
            )
            nodes.append(node)
        return nodes
    
    
    
    
    def point2Mat(self,savepath):
        """保存点位信息图"""
        image=np.zeros((self.gh,self.gw,3),dtype=np.uint8)
        for point_idx, point in enumerate(self.points):
            rect=point["box"]
            if len(rect) == 4:
                cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255, 255, 255), 1)
            if len(rect)>4:
                pts = np.array(rect).reshape(-1, 2).astype(np.int32)
                cv2.polylines(image, [pts], False, (255, 255, 255), 1)
        cv2_imwrite_chinese(savepath,image)
        return image  


    def export_node_to_labelme(
        self,node,
        image_path: str,
        save_path: str,
        encoding: str = "utf-8"
    ) -> bool:
        """
        将all_point（self.points）中的点转换为LabelMe格式JSON文件（带坐标变换）
        输出：矩形标注（rectangle）
        """
        image=cv2_imread_chinese(image_path)
        M=jx_M(image,self.size)
        try:
            labelme_data = {
                "version": "4.2.10",
                "flags": {},
                "shapes": [],
                "imagePath": os.path.basename(image_path),
                "imageData": None,
                "imageHeight": self.size[1],
                "imageWidth": self.size[0]
            }

            for point_idx, point in enumerate(node["points"]):
                cimetypes=point["cimetypes"]
                cimeid=point["cimeid"]
                cimename=point["cimename"]
                rect=point["cimeboxs"]
                transformed_coords = M_jsuan(rect, M)
                if not transformed_coords:
                    continue
                
                # 自动区分矩形/折线
                if len(transformed_coords) == 4:
                    # 矩形逻辑（完全保留你原有的使用方式）
                    x1, y1, x2, y2 = transformed_coords
                    shape = {
                        "label": cimename+"_"+cimetypes+"_"+cimeid,
                        "points": [
                            [x1, y1],       # 左上角
                            [x2, y2]        # 右下角
                        ],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",  # 矩形类型
                        "flags": {}
                    }
                    labelme_data["shapes"].append(shape)

            # 新增：将 ssnodes 中 buttons 的 box 也保存为 LabelMe 矩形标注
            for button in node.get("buttons", []):
                box = button.get("box")
                if not box or len(box) != 4:
                    continue
                transformed_coords = M_jsuan(box, M)
                if not transformed_coords or len(transformed_coords) != 4:
                    continue
                x1, y1, x2, y2 = transformed_coords
                shape = {
                    "label": "button_" + str(button.get("type", "")),
                    "points": [
                        [x1, y1],
                        [x2, y2]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                }
                labelme_data["shapes"].append(shape)

            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_path, "w", encoding=encoding) as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=4)
            print(f"成功导出 {len(labelme_data['shapes'])} 个矩形标注到 LabelMe：{save_path}")
            return True
        except Exception as e:
            print(f"导出LabelMe失败：{e}")
            return False

# ===================== 6. 节点去重与场景图构建 =====================
def deduplicate_points(points_list: list) -> list:
    """点位去重"""
    seen = set()
    deduplicated = []
    for point in points_list:
        cimeid = point.get("cimeid", "")
        ycyx = point.get("ycyx", "")
        unique_key = (cimeid, ycyx)
        if cimeid and unique_key not in seen:
            seen.add(unique_key)
            deduplicated.append(point)
    logger.info(f"点去重完成：原始{len(points_list)}个 → 去重后{len(deduplicated)}个")
    return deduplicated

def build_scene_graph(nodes):
    """创建map节点图"""
    scene_graph = {}
    for node in nodes:
        scene_name = node["name"]
        children_names = node.get("children_names", [])
        buttons = node.get("buttons", [])
        points = node.get("points", [])
        child_jump_map = {}
        max_len = max(len(children_names), len(buttons))
        for i in range(max_len):
            child_name = children_names[i] if i < len(children_names) else ""
            button = buttons[i] if i < len(buttons) else {}
            if child_name:
                child_jump_map[child_name] = button
        scene_graph[scene_name] = {
            "children": child_jump_map,
            "points": points,
            "children_names": children_names,
            "buttons": buttons
        }
    return scene_graph

def find_point_belong_scene(scene_graph, point_id, mode="yx"):
    """查找点位所属节点"""
    scene_names = []
    for scene_name, scene_info in scene_graph.items():
        for point in scene_info["points"]:
            if isinstance(point, dict) and point.get("cimeid") == point_id and point.get("ycyx") == mode:
                scene_names.append(scene_name)
    return scene_names

def find_jump_path_single_pair(scene_graph, start_scene, end_scenes, visited_scenes):
    if visited_scenes is None:
        visited_scenes = set()
    queue = deque()
    queue.append((start_scene, []))
    visited_scenes.add(start_scene)
    while queue:
        current_scene, current_path = queue.popleft()
        scene_info = scene_graph.get(current_scene, {})
        child_jump_map = scene_info.get("children", {})
        for child_scene, button in child_jump_map.items():
            if child_scene in end_scenes:
                current_path.append({"from_scene": current_scene, "to_scene": child_scene, "button": button})
                return current_path
            if child_scene not in visited_scenes:
                visited_scenes.add(child_scene)
                new_path = current_path.copy()
                new_path.append({"from_scene": current_scene, "to_scene": child_scene, "button": button})
                queue.append((child_scene, new_path))
    return None

def find_jump_from_node2(scene_graph, start_node, end_point_id, end_point_ycyx, came_from_node=None):
    end_scenes = find_point_belong_scene(scene_graph, end_point_id, end_point_ycyx)
    if not end_scenes:
        logger.error(f"终点 {end_point_id} 未找到归属场景")
        return []
    if start_node in end_scenes:
        return []
    queue = deque()
    queue.append((start_node, came_from_node, []))
    visited = set()
    while queue:
        current, prev, path = queue.popleft()
        data = scene_graph.get(current, {})
        child_jump_map = data.get("children", {})
        for target_scene, button in child_jump_map.items():
            target = target_scene
            if target_scene == "BACK" and prev:
                target = prev
            state = (current, target)
            if state in visited:
                continue
            visited.add(state)
            new_step = {"from_scene": current, "to_scene": target, "button": button}
            new_path = path + [new_step]
            if target in end_scenes:
                return new_path
            queue.append((target, current, new_path))
    logger.warning("未找到跳转路径")
    return []
def find_jump_from_node3(scene_graph, start_node, end_scenes,came_from_node=None):
    """查找点位跳转路径"""
    if start_node in end_scenes:
        return []
    queue = deque()
    queue.append((start_node, came_from_node, []))
    visited = set()
    while queue:
        current, prev, path = queue.popleft()
        data = scene_graph.get(current, {})
        child_jump_map = data.get("children", {})
        for target_scene, button in child_jump_map.items():
            target = target_scene
            if target_scene == "BACK" and prev:
                target = prev
            state = (current, target)
            if state in visited:
                continue
            visited.add(state)
            new_step = {"from_scene": current, "to_scene": target, "button": button}
            new_path = path + [new_step]
            if target in end_scenes:
                return new_path
            queue.append((target, current, new_path))
    logger.warning("未找到跳转路径")
    return []

# ===================== 7. 递归构建节点树（零重复初始化） =====================
def FromGstoMap(global_index: GlobalDataIndex, jgdir: str, g_path: str, button_config: dict = None):
    """有光字牌分页情况"""
    all_nodes = []
    processed_names = set()
    nodename = os.path.basename(g_path)
    button_config = button_config or {}

    def recursive_build_nodes(current_g_path, nodename, parent_node_name=None):
        print(f'current_g_path:{current_g_path}, nodename:{nodename}, parent_node_name:{parent_node_name}')
        base_g_name = os.path.basename(current_g_path)
        # if(base_g_name=="SH.新金阳站河阳4178间隔接线图.bay.pic.g"):
        #     print(1)
        if base_g_name in processed_names:
            logger.info(f"节点【{nodename}】已处理，跳过递归")
            return
        processed_names.add(base_g_name)

        try:
            # 仅初始化一次AnalyseG2，复用全局索引，零IO
            rootg = AnalyseG2(current_g_path, global_index, button_config)
            rootg.getallpoint()

            if "bay.pic.g" in base_g_name:
                actual_parent_name = parent_node_name if parent_node_name else "根节点"
                ssnodes = rootg.CreatJGMapbay(nodename, actual_parent_name)
                for ssn_node in ssnodes:
                    ssn_node["points"] = deduplicate_points(ssn_node["points"])
                all_nodes.extend(ssnodes)
                # 递归子节点
                for ssn_node in ssnodes:
                    for child_name in ssn_node.get("children_names", []):
                        if child_name in processed_names:
                            continue
                        child_path = global_index.find_file(child_name)
                        if child_path:
                            recursive_build_nodes(child_path, child_name, nodename)
            else:
                if rootg.has_gzp_navigation():
                    node = rootg.CreatGZPMap(base_g_name)
                else:
                    node = rootg.CreatMap(base_g_name)
                # 全局点去重
                filtered_points = []
                for point in node.get("points", []):
                    cimeid = point.get("cimeid", "")
                    ycyx = point.get("ycyx", "")
                    filtered_points.append(point)
                node["points"] = filtered_points
                children = node.get("children_names", [])
                for i in range(len(children)):
                    child_name = children[i]
                    if child_name in processed_names:
                        continue
                    if ".g" not in child_name:
                        children[i] = os.path.basename(g_path)
                node["children_names"] = children
                all_nodes.append(node)
                # 递归子节点
                for child_name in node.get("children_names", []):
                    if child_name in processed_names:
                        continue
                    child_path = global_index.find_file(child_name)
                    if child_path:
                        recursive_build_nodes(child_path, child_name, nodename)
        except Exception as e:
           logger.exception(f"处理G文件【{current_g_path}】失败")

    recursive_build_nodes(g_path, nodename, parent_node_name=None)
    return all_nodes

# ===================== 7. 递归构建节点树（零重复初始化） =====================
def FromGstoMapSGZ(global_index: GlobalDataIndex, jgdir: str, g_path: str, button_config: dict = None):
    """无光字牌分页情况"""
    all_nodes = []
    processed_names = set()
    nodename = os.path.basename(g_path)
    button_config = button_config or {}

    def recursive_build_nodes(current_g_path, nodename, parent_node_name=None):
        base_g_name = os.path.basename(current_g_path)
        if base_g_name in processed_names:
            logger.info(f"节点【{nodename}】已处理，跳过递归")
            return
        processed_names.add(base_g_name)

        try:
            # 仅初始化一次AnalyseG2，复用全局索引，零IO
            rootg = AnalyseG2(current_g_path, global_index, button_config)
            rootg.getallpoint()

            
            node = rootg.CreatMap(base_g_name)
            # 全局点去重
            filtered_points = []
            for point in node.get("points", []):
                cimeid = point.get("cimeid", "")
                ycyx = point.get("ycyx", "")
                filtered_points.append(point)
            node["points"] = filtered_points
            children = node.get("children_names", [])
            for i in range(len(children)):
                child_name = children[i]
                if child_name in processed_names:
                    continue
                if ".g" not in child_name:
                    children[i] = os.path.basename(g_path)
            node["children_names"] = children
            all_nodes.append(node)
            # 递归子节点
            for child_name in node.get("children_names", []):
                if child_name in processed_names:
                    continue
                child_path = global_index.find_file(child_name)
                if child_path:
                    recursive_build_nodes(child_path, child_name, nodename)
        except Exception as e:
           logger.exception(f"处理G文件【{current_g_path}】失败")

    recursive_build_nodes(g_path, nodename, parent_node_name=None)
    return all_nodes



def save_nodes_to_json(nodes, save_path, indent=4, encoding="utf-8"):
    """保存节点树结构"""
    try:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, "w", encoding=encoding) as f:
            json.dump(nodes, f, ensure_ascii=False, indent=indent)
        logger.info(f"节点列表已保存到：{save_path}")
        return True
    except Exception as e:
        logger.exception(f"保存JSON失败")
        return False

# ===================== 8. 主程序入口 =====================
if __name__ == "__main__":
    # 配置参数
    # 配置参数
    test_station = "上海.金阳"
    test_cime = "/workspace/sortandcheck/jyz2/js_sh_jy_202606181519.CIME"
    test_g = "/workspace/sortandcheck/jyz2/jg"
    test_main_g = "/workspace/sortandcheck/jyz2/SH.新金阳站总图.fac.pic.g"
    test_yx = ["/workspace/sortandcheck/jyz2/yx/jy_yx.txt"]   # 请替换为实际遥信点表路径
    test_yc = ["/workspace/sortandcheck/jyz2/yc/jy_yc.txt"]  # 请替换为实际遥测点表路径

    tyyuandir="/workspace/sortandcheck/tuyuan"
    # ========== 核心流程（多表场景下极速运行） ==========
    # 1. 一次性初始化全局索引，解析所有表，构建O(1)索引（仅执行1次）
    global_index = GlobalDataIndex(test_station, test_cime, test_yc,test_yx,output_dir="./csv_modules")
    # 2. 一次性构建文件路径索引，彻底消灭重复os.walk（仅执行1次）
    global_index.build_file_path_index(root_dirs=[test_g, tyyuandir])
    # 3. 递归构建所有节点，全程复用全局索引，零重复IO
    # button_config = {"cy_事故.gzp.icn.g": False}
    button_config={}
    all_nodes = FromGstoMap(global_index, test_g, test_main_g, button_config)
    save_nodes_to_json(all_nodes, "./maps/金阳站.json")
    # # 4. 构建场景跳转图
    scene_graph = build_scene_graph(all_nodes)
    print(all_nodes)
    rootg = AnalyseG2(test_main_g, global_index, button_config)
    rootg.getallpoint()
    # ssnodes = rootg.CreatJGMapbay("jg", "back")
    ssnodes = all_nodes[0]
    # ssnodes = rootg.CreatMap(test_station)
    rootg.export_node_to_labelme(ssnodes,r"/workspace/sortandcheck/jyz2/jy5.png",save_path="/workspace/sortandcheck/jyz2/jy5.json")
    # 5. 查找跳转路径
    # path = find_jump_from_node3(
    #     scene_graph,
    #     start_node="SH.新金阳站河阳4178间隔接线图.bay.pic.g",
    #     end_scenes=["SH.新金阳站35KV一次接线图.fac.pic.g"],
    #     came_from_node="SH.新金阳站总图.fac.pic.g"
    # )
    # print("跳转路径：", path)
    # # 6. 保存结果
    
