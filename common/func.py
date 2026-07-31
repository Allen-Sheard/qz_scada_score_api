import re
import cv2
import math
import numpy as np
import cv2
import re
from collections import Counter
import itertools
def is_name_strict_match( base_name: str, target_name: str) -> bool:
    """
    精准逻辑：
    1. 找到 base_name 在 target_name 中的位置
    2. 检查它后面【紧跟】的是不是数字
    3. 后面不是数字 → 返回 True
    """
    if not base_name or not target_name:
        return False

    # 1. 定位 base_name 出现的位置
    idx = target_name.find(base_name)
    if idx == -1:
        return False  # 根本没出现

    # 2. 计算 base_name 结束后的下一个字符位置
    next_pos = idx + len(base_name)

    # 3. 如果已经是字符串末尾 → 合格
    if next_pos >= len(target_name):
        return True

    # 4. 检查【紧跟后面的字符】是不是数字
    next_char = target_name[next_pos]
    if next_char.isdigit():
        return False  # 后面是数字 → 不合格

    # 5. 后面不是数字 → 合格
    return True
def is_string_valid(s):
    """
    校验字符串合法性：
    1. 含「一段/二段/三段/四段」→ 直接合法
    2. 含数字：提取连续数字，检查其后2个字符是否含「千伏/kv/KV」，
       若找到无千伏后缀的连续数字 → 合法；否则非法
    :param s: 待校验字符串
    :return: bool - 合法返回True，非法返回False
    """
    # 预处理：去除所有空格（避免空格干扰判断）
    s_processed = s.replace(" ", "").replace("　", "")
    if not s_processed:
        return False  # 空字符串直接非法
    
    # 规则1：含「一段/二段/三段/四段」→ 直接合法
    section_keywords = ["一段", "二段", "三段", "四段","段"]
    if any(keyword in s_processed for keyword in section_keywords):
        return True
    
    # 规则2：处理数字+千伏的逻辑
    # 步骤1：提取所有连续数字的位置和内容（正则匹配）
    # 正则匹配连续数字，返回 (起始索引, 结束索引, 数字字符串)
    num_matches = []
    for match in re.finditer(r'\d+', s_processed):
        num_start = match.start()
        num_end = match.end()
        num_str = match.group()
        num_matches.append((num_start, num_end, num_str))
    chinese_num_pattern = r'[零一二三四五六七八九十]+'
    for match in re.finditer(chinese_num_pattern, s_processed):
        num_start = match.start()
        num_end = match.end()
        num_str = match.group()
        num_matches.append((num_start, num_end, num_str))
    if not num_matches:
        return False  # 无数字且无段关键词 → 非法
    
    # 步骤2：遍历每个连续数字，检查其后2个字符是否含千伏相关关键词
    kv_keywords = ["千伏", "kv", "KV","kV","Kv"]
    for (num_start, num_end, num_str) in num_matches:
        # 提取数字后2个字符（避免越界）
        check_end = min(num_end + 2, len(s_processed))
        after_chars = s_processed[num_end:check_end]
        
        # 检查后2个字符是否含千伏关键词
        has_kv = any(kv in after_chars for kv in kv_keywords)
        
        # 找到无千伏后缀的数字 → 合法
        if not has_kv:
            return True
    
    # 所有数字后都有千伏后缀 → 非法
    return False

def extract_unique_longest_common_patterns(str_list, min_len=2):
    """
    提取「最长不被包含的核心公共子串」：如果子串已被更长的公共子串包含，则丢弃
    :param str_list: 字符串数组（允许重复、空值）
    :param min_len: 保留的最小子串长度
    :return: 干净的最长核心公共子串列表
    """
    # 步骤1：输入去重 + 过滤空字符串
    unique_strs = list(set(str_list))
    valid_strs = [s for s in unique_strs if s.strip() != ""]
    if(len(valid_strs)==1):
        return valid_strs
    if len(valid_strs) < 2:
        return []
    
    # 步骤2：以最短字符串为基准
    valid_strs_sorted = sorted(valid_strs, key=lambda x: len(x))
    base_str = valid_strs_sorted[0]
    other_strs = valid_strs_sorted[1:]
    
    # 步骤3：生成所有可能子串并筛选公共子串
    all_substrings = set()
    n = len(base_str)
    for i in range(n):
        for j in range(i + 1, n + 1):
            all_substrings.add(base_str[i:j])
    
    common_substrings = []
    for substr in all_substrings:
        if len(substr) < min_len:
            continue
        found_in_all = True
        for s in other_strs:
            if substr not in s:
                found_in_all = False
                break
        if found_in_all:
            common_substrings.append(substr)
    
    # 步骤4：【核心逻辑】只保留「不被其他任何公共子串包含」的最长片段
    # 先按长度从长到短排序
    common_substrings.sort(key=lambda x: (-len(x), x))
    
    unique_longest = []
    seen_patterns = set()
    
    for substr in common_substrings:
        # 检查当前子串是否已经被包含在之前保留的更长的子串里
        is_subsumed = False
        for seen in seen_patterns:
            if substr in seen:
                is_subsumed = True
                break
        if not is_subsumed:
            unique_longest.append(substr)
            seen_patterns.add(substr)
    
    return unique_longest


def clean_name(s: str) -> str:
    """
    清理字符串：
    1. 删除 , . / ( ) 这些符号
    2. 所有字母小写
    """
    if not s:
        return ""
    
    # 1. 转小写
    s = s.lower()
    
    # 2. 移除所有 , . / ( )

    s = re.sub(r'[^a-z0-9\u4e00-\u9fa5]', '', s)
    
    # 3. 清理多余空格（可选，建议加上）
    s = s.strip()
    
    return s
def is_exact_match(a: list, b: list) -> bool:
    """
    终极严格匹配：
    1. 长度必须一样
    2. 每个位置必须一一对应
    3. 每个位置内部的【子列表】也必须完全一样
    4. 顺序不一样 → 不匹配
    """
    # 长度不同直接不匹配
    if len(a) != len(b):
        return False

    # 逐个位置对比
    for item_a, item_b in zip(a, b):
        # 如果是列表 → 递归对比
        if isinstance(item_a, list) and isinstance(item_b, list):
            if not is_exact_match(item_a, item_b):
                return False
        # 如果是普通元素 → 直接对比
        else:
            if item_a != item_b:
                return False

    # 全部位置都完全一致
    return True

def cv2_imwrite_chinese(path, img):
    try:
        # 核心：直接编码 + 二进制写入，支持中文、最简单、最稳
        _, img_encode = cv2.imencode('.jpg', img)
        with open(path, 'wb') as f:
            f.write(img_encode)
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False
    


# ------------------- 【核心新增】线段共线重叠判断 -------------------
def is_segments_colinear_overlap(seg1, seg2):
    """
    判断两条线段是否共线且有重叠（重合）
    参数：
        seg1/seg2: 线段格式 [(x1,y1), (x2,y2)]
    返回：
        bool: True=共线且有重叠，False=不重合
    """
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    # 1. 判断是否共线（叉积为0）
    def cross(xa, ya, xb, yb):
        return xa * yb - ya * xb
    
    # 向量1、向量2、向量1到向量2起点的向量
    v1 = (x2 - x1, y2 - y1)
    v2 = (x4 - x3, y4 - y3)
    v13 = (x3 - x1, y3 - y1)
    
    # 不共线直接返回False
    if abs(cross(v1[0], v1[1], v2[0], v2[1])) > 1e-10:
        return False
    # 共线但v13不共线，也返回False
    if abs(cross(v1[0], v1[1], v13[0], v13[1])) > 1e-10:
        return False

    # 2. 判断线段是否有重叠（投影到坐标轴判断范围）
    def overlap(min1, max1, min2, max2):
        return max(min1, min2) <= min(max1, max2) + 1e-10
    
    # 投影到x轴
    min1_x, max1_x = min(x1, x2), max(x1, x2)
    min2_x, max2_x = min(x3, x4), max(x3, x4)
    # 投影到y轴
    min1_y, max1_y = min(y1, y2), max(y1, y2)
    min2_y, max2_y = min(y3, y4), max(y3, y4)
    
    # x和y轴投影都有重叠，才是线段重合
    return overlap(min1_x, max1_x, min2_x, max2_x) and overlap(min1_y, max1_y, min2_y, max2_y)



def get_segment_intersection(seg1, seg2):
    """计算两条线段的唯一交点，无交点/共线返回None"""
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    
    # 计算分母
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # 平行/共线，无唯一交点
    
    # 计算交点参数
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    s = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    
    # 交点必须在线段上（允许微小浮点误差）
    if -1e-10 <= t <= 1.0 + 1e-10 and -1e-10 <= s <= 1.0 + 1e-10:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    
    return None


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    px_dx = px - x1
    py_dy = py - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-10:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, (px_dx * dx + py_dy * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def rect_to_segments(rect_points):
    (x1,y1),(x2,y2)=rect_points; p1,p2,p3,p4=(min(x1,x2),min(y1,y2)),(max(x1,x2),min(y1,y2)),(max(x1,x2),max(y1,y2)),(min(x1,x2),max(y1,y2))
    return [[p1,p2],[p2,p3],[p3,p4],[p4,p1]]

def get_text_features(obj):
    """
    仅提取必需的文本特征
    ✅ 完美兼容：
       - 一维格式：[x1, y1, x2, y2, x3, y3...]
       - 二维格式：[[x1,y1], [x2,y2], ...]
    自动计算所有点的最小外接矩形
    """
    points = np.array(obj['points'])

    # ============= 核心兼容逻辑 =============
    if points.ndim == 1:
        # 一维：[x1,y1,x2,y2...] → 转成二维 [[x1,y1],[x2,y2]]
        xs = points[0::2]  # 取偶数位：x
        ys = points[1::2]  # 取奇数位：y
    else:
        # 二维：[[x,y], ...]
        xs = points[:, 0]
        ys = points[:, 1]

    # 统一计算最小外接矩形
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    width = x_max - x_min
    height = y_max - y_min
    aspect_ratio = height / width if width > 0 else 0

    return {
        "top_left": (x_min, y_min),
        "bottom_left": (x_min, y_max),
        "bottom_right": (x_max, y_max),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "y_center": (y_min + y_max) / 2,
        "x_center": (x_min + x_max) / 2
    }

def get_text_type(text):
    if not text:
        return 2  # 空
    has_pure_en = bool(re.match(r'^[a-zA-Z]+$', text))
    has_cn = bool(re.search(r'[\u4e00-\u9fff]', text))
    if has_pure_en and not has_cn:
        return 1  # 纯英文
    else:
        return 2  # 混合
    
def parse_points(points):
    arr = np.array(points).flatten()
    # 按 x,y 分组，提取所有端点
    return [(arr[i], arr[i+1]) for i in range(0, len(arr), 2)]

def remove_duplicate_lists(target_dict):
    """
    清理字典中每个key下的重复列表项
    :param target_dict: 格式为 {key: [list1, list2, ...]} 的字典
    :return: 去重后的字典
    """
    cleaned_dict = {}
    for key, list_of_lists in target_dict.items():
        # 方法：将列表转为元组（可哈希），用集合去重，再转回列表
        seen = set()
        unique_lists = []
        for sub_list in list_of_lists:
            # 递归处理嵌套列表 → 转为嵌套元组
            def list_to_tuple(nested_list):
                if isinstance(nested_list, list):
                    return tuple(list_to_tuple(item) for item in nested_list)
                return nested_list
            
            tuple_version = list_to_tuple(sub_list)
            if tuple_version not in seen:
                seen.add(tuple_version)
                unique_lists.append(sub_list)
        cleaned_dict[key] = unique_lists
        print(f"tpkeyid {key} 去重前：{len(list_of_lists)}条 → 去重后：{len(unique_lists)}条")
    return cleaned_dict

def get_intersection_area(box1, box2):
    # 解包坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算相交区域
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    # 没有相交 → 面积0
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0
    # 相交面积
    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

def CheckINOneStation(points,saveimagepath,filename):
    error_list=[]
    sta_list=[]
    for i in range(len(points)):
        tree=points[i]
        if tree["lx"].lower() in ["dtext","protect", "grounddisconnector", "disconnector", "cbreaker", "gzp","dollybreaker"]:
            cimeid_list=tree.get("cimeid", [])
            if not cimeid_list or "" in cimeid_list:
                oneerror = {
                            "level": 1,
                            "msg": f"来自{filename}的错误：缺少104点号",
                            "detail": {
                                "type": 1,
                                "cime_id": [],
                                "data": {
                                    "x1":tree["box"][0],
                                    "y1":tree["box"][1],
                                    "x2":tree["box"][2],
                                    "y2":tree["box"][3]
                                },
                                "img": saveimagepath
                            }
                            }
                error_list.append(oneerror)
            else:
                stanames=tree.get("站名",[])
                if stanames:
                    sta_list.extend(stanames)
    correct_sta_name = None
    if sta_list:  # 确保有站名数据才统计
        sta_count = Counter(sta_list)
        # most_common(1) 返回 [(站名, 次数)]，取第一个元素的站名
        correct_sta_name = sta_count.most_common(1)[0][0]
        for tree in points:
            if tree["lx"].lower() in ["dtext","protect", "grounddisconnector", "disconnector", "cbreaker", "gzp","dollybreaker"] and tree["cimeid"] != []:
                stanames = tree.get("站名", [])
                for sta_name in stanames:
                    if sta_name:
                        if sta_name != correct_sta_name:
                            # 错误站名，添加错误信息
                            oneerror = {
                            "level": 1,
                            "msg": f"来自{filename}的错误：当前站名应为：{correct_sta_name},存在其他站的点位：{sta_name}",
                            "detail": {
                                "type": 1,
                                "cime_id": [],
                                "data": {
                                    "x1":tree["box"][0],
                                    "y1":tree["box"][1],
                                    "x2":tree["box"][2],
                                    "y2":tree["box"][3]
                                },
                                "img": saveimagepath
                            }
                            }
                            error_list.append(oneerror)
    return error_list

def CheckGZPandText(points,saveimagepath,filename):
    error_list=[]
    for i in range(len(points)):
        treegzp=points[i]
        if(treegzp["cimeid"]==[]):
            continue
        if(treegzp["cimeid"][0]==""):
            continue
        gzpname=clean_name(treegzp["cimename"][0])
        gzpbox=treegzp["box"]
        allow_text=None
        bh_text=None
        if treegzp["lx"].lower()=="gzp":
            for j in range(len(points)):
                treetext=points[j]
                if treetext["lx"].lower()=="text":
                    txtname = clean_name(treetext["name"])
                    if gzpname in txtname or txtname in gzpname:
                        allow_text=treetext
                        break
            max_area = 0    
            for j in range(len(points)):
                treetext = points[j]
                if treetext["lx"].lower() == "text":
                    txtbox = treetext["box"]
                    # 计算相交面积
                    area = get_intersection_area(gzpbox, txtbox)
                    # 记录面积最大的
                    if area > max_area:
                        max_area = area
                        bh_text = treetext
        if allow_text is not None and bh_text is not None:
            if allow_text == bh_text:
                pass
            else:
                oneerror = {
                            "level": 1,
                            "msg": f"来自{filename}的错误：{allow_text['name']} 位置与光字牌的位置不靠近或者存在其他文字区域覆盖在光字牌上面",
                            "detail": {
                                "type": 1,
                                "cime_id": [],
                                "data": {
                                    "x1": gzpbox[0],
                                    "y1": gzpbox[1],
                                    "x2": gzpbox[2],
                                    "y2": gzpbox[3]
                                },
                                "img": saveimagepath
                            }
                            }
                
                error_list.append(oneerror)

        # 2. allow_text 为空，但 bh_text 存在 → 报错
        elif allow_text is None and bh_text is not None:
            oneerror = {
                            "level": 1,
                            "msg": f"来自{filename}的错误：存在相交文本{bh_text['name']}，但是与实际点位名称{gzpname}存在差异",
                            "detail": {
                                "type": 1,
                                "cime_id": [],
                                "data": {
                                    "x1": gzpbox[0],
                                    "y1": gzpbox[1],
                                    "x2": gzpbox[2],
                                    "y2": gzpbox[3]
                                },
                                "img": saveimagepath
                            }
                            }
            error_list.append(oneerror)

        # 3. 两个都为空 → 没有找到对应的 text
        elif allow_text is None and bh_text is None:
            oneerror = {
                            "level": 1,
                            "msg": f"来自{filename}的错误：{gzpname}光字牌没有找到对应的解释文本",
                            "detail": {
                                "type": 1,
                                "cime_id": [],
                                "data": {
                                    "x1": gzpbox[0],
                                    "y1": gzpbox[1],
                                    "x2": gzpbox[2],
                                    "y2": gzpbox[3]
                                },
                                "img": saveimagepath
                            }
                            }
            error_list.append(oneerror)

        # 4. bh_text 为空，allow_text 不为空 → 判断是否同行/同列，距离≤100
        elif bh_text is None and allow_text is not None:
            gx1, gy1, gx2, gy2 = gzpbox
            tx1, ty1, tx2, ty2 = allow_text["box"]

            # 中心坐标
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2
            tcx = (tx1 + tx2) / 2
            tcy = (ty1 + ty2) / 2

            dx = abs(gcx - tcx)
            dy = abs(gcy - tcy)

            same_row = dy < ((gy2- gy1) / 2.0)  # 近似同一行
            same_col = dx < ((gx2- gx1) / 2.0)  # 近似同一列
            distance_ok = (dx <= 100 and same_row) or (dy <= 100 and same_col)

            if not distance_ok:
                oneerror = {
                            "level": 1,
                            "msg": f"来自{filename}的错误：{gzpname}文本与gzp不在同一行/列，或距离过远",
                            "detail": {
                                "type": 1,
                                "cime_id": [],
                                "data": {
                                    "x1": gzpbox[0],
                                    "y1": gzpbox[1],
                                    "x2": gzpbox[2],
                                    "y2": gzpbox[3]
                                },
                                "img": saveimagepath
                            }
                            }
                error_list.append(oneerror)
    return error_list

def GetTpname(points, filename, whitelist=("千伏一(", "千伏二(")):
    """
    优化版：提取公共前缀，自动剔除异常名称，公共字段后不能是纯数字
    保留原有逻辑，消除重复代码，修复边界bug
    """
    # ===================== 工具函数：提取并验证前缀 =====================
    def check_prefix(test_names):
        """内部复用：提取公共前缀 + 验证规则"""
        prefixs = extract_unique_longest_common_patterns(test_names)
        # 找第一个合法前缀
        current_prefix = next((p for p in prefixs if is_string_valid(p)), "")
        if not current_prefix:
            return None

        # 白名单优先命中，直接返回
        if current_prefix.endswith(whitelist):
            return current_prefix

        # 规则：公共字段后面 不可以是纯数字
        for name in test_names:
            idx = name.find(current_prefix)
            if idx == -1:
                return None

            pos = idx + len(current_prefix)
            suffix1 = name[pos:pos+1].strip()
            suffix2 = name[pos:pos+2].strip()
            prev_char = name[pos-1:pos]  # 前缀最后一个字符

            # 原规则：后缀是数字 + 前一个也是数字 ｜｜ 后缀是“千伏” → 无效
            if (suffix1.isdigit() and prev_char.isdigit()) or (suffix2 == "千伏"):
                return None

        return current_prefix

    # ===================== 第一步：收集所有 name =====================
    name_parts = [clean_name(filename)]
    for p in points:
        for name in p.get("cimename", []):
            if name:
                name_parts.append(clean_name(name))

    # ===================== 策略：依次尝试删除 0~6 个名称 =====================
    best_prefix = ""
    for remove_count in range(0, 7):  # 0,1,2,3,4,5,6
        if remove_count == 0:
            # 不删除任何名称
            best_prefix = check_prefix(name_parts)
            if best_prefix:
                break
        else:
            # 尝试删除 N 个名称，找有效组合
            for to_remove in itertools.combinations(name_parts, remove_count):
                test_names = [n for n in name_parts if n not in to_remove]
                res = check_prefix(test_names)
                if res:
                    best_prefix = res
                    return best_prefix  # 找到立刻返回，更快

    return best_prefix

def CheckINOnetp(points,saveimagepath,filename):
    error_list=[]
    tpname=GetTpname(points,filename)
    if(tpname==""):
        oneerror = {
                        "level": 1,
                        "msg": f"来自{filename}的错误：没有找到合适间隔名称",
                        "detail": {
                            "type": 2,
                            "cime_id": [],
                            "data": {},
                            "img": saveimagepath
                        }
                    }
        error_list.append(oneerror)
        return error_list,tpname
    if tpname not in clean_name(filename):
        oneerror = {
                        "level": 1,
                        "msg": f"来自{filename}的错误：{tpname}间隔名称与G文件间隔名称不匹配",
                        "detail": {
                            "type": 2,
                            "cime_id": [],
                            "data": {},
                            "img": saveimagepath
                        }
                    }
        error_list.append(oneerror)
    for p in points:
        for name in p.get("cimename", []):
            if not name:
                continue
            idx = name.find(tpname)
            if idx == -1:
                box=points["box"]
                oneerror = {
                        "level": 1,
                        "msg": f"来自{filename}的错误：{tpname}间隔名称点位名称{name}不在同一间隔",
                        "detail": {
                            "type": 1,
                            "cime_id": [],
                            "data": {
                                "x1":box[0],
                                "y1":box[1],
                                "x2":box[2],
                                "y2":box[3]
                            },
                            "img": saveimagepath
                        }
                    }
                error_list.append(oneerror)
    return error_list,tpname

def CheckDtextTextTuple(points,saveimagepath,filename,TextTuple={}):
    return []


                    

