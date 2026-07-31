
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO  # YOLO11核心库
import cv2
class YOLO11DigitDetector:
    """
    YOLO11数字检测+数字组合+转换完整处理器
    特性：
    1. 初始化时仅加载一次模型（节省资源）
    2. 输入单张图片 → 直接返回数字/失败结果
    3. 内置所有数字处理规则（过滤/去重/特殊字符/转换）
    """
    def __init__(self, model_path: str, image_height_threshold: int = 0, 
                 iou_threshold_normal: float = 0.7, iou_threshold_1: float = 0.9):
        """
        初始化检测器（仅加载一次模型）
        :param model_path: YOLO11模型文件路径（.pt文件）
        :param image_height_threshold: 数字框高度阈值（<=0时自动取图片高度的一半）
        :param iou_threshold_normal: 普通数字去重IOU阈值
        :param iou_threshold_1: 数字1去重IOU阈值（更宽松）
        """
        # 1. 加载YOLO11模型（仅初始化一次）
        print(f"正在加载YOLO11模型: {model_path}")
        self.model = YOLO(model_path)
        print("模型加载完成！")
        
        # 2. 初始化参数
        self.image_height_threshold = image_height_threshold
        self.iou_threshold_normal = iou_threshold_normal
        self.iou_threshold_1 = iou_threshold_1
        # 类别映射（YOLO11输出的class id对应数字/符号）
        # 请根据你的训练数据集调整此映射！
        # 示例：id 0='-', 1='0', 2='1', ..., 10='9', 11='.'
        self.class_id_to_char = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: '.',
            11: '-'
        }

    def calculate_iou_xaxis(self, box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        """仅计算x轴方向的交并比（IOU），用于数字去重"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        inter_x1 = max(x1_1, x1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_width = max(0, inter_x2 - inter_x1)
        union_width = max(x2_1, x2_2) - min(x1_1, x1_2)
        
        return inter_width / union_width if union_width != 0 else 0.0

    def filter_small_digits(self, detections: List[Dict], image_height: int) -> List[Dict]:
        """过滤高度不符合要求的数字框"""
        # 确定高度阈值：用户指定值 或 图片高度的一半
        threshold = self.image_height_threshold if self.image_height_threshold > 0 else image_height / 2
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            current_char = det['char']
            if(current_char==".")or(current_char=="-"):
                filtered.append(det)
                continue
            if (y2 - y1) >= threshold:
                filtered.append(det)
        
        return filtered

    def process_special_chars(self, detections: List[Dict], image_height: int) -> List[Dict]:
        """处理特殊字符（负号、小数点），严格遵循所有规则"""
        detections_sorted = sorted(detections, key=lambda d: (d['box'][0] + d['box'][2])/2)
        negatives = [d for d in detections_sorted if d['char'] == '-']
        decimals = [d for d in detections_sorted if d['char'] == '.']
        digits = [d for d in detections_sorted if d['char'] not in ['-', '.']]
        
        processed = []
        
        # 处理负号：仅保留最左侧第一个
        if negatives and len(processed) == 0:
            processed.append(negatives[0])
        
        # 添加普通数字
        processed.extend(digits)
        
        # 处理小数点：底部(70%高度以下)+非最左+非最右+仅保留一个
        bottom_decimals = [d for d in decimals if d['box'][3] >= image_height * 0.5]
        if bottom_decimals:
            leftmost_dec = min(bottom_decimals, key=lambda d: (d['box'][0] + d['box'][2])/2)
            dec_center_x = (leftmost_dec['box'][0] + leftmost_dec['box'][2])/2
            
            # 计算所有字符的x轴中心范围
            all_centers = [(d['box'][0] + d['box'][2])/2 for d in processed] + [dec_center_x]
            min_x, max_x = min(all_centers), max(all_centers)
            
            # 校验小数点位置
            if dec_center_x != min_x and dec_center_x != max_x:
                # 插入到正确位置
                insert_idx = 0
                for i, d in enumerate(processed):
                    if (d['box'][0] + d['box'][2])/2 > dec_center_x:
                        insert_idx = i
                        break
                processed.insert(insert_idx, leftmost_dec)
        
        return processed

    def remove_duplicate_digits(self, detections: List[Dict]) -> List[Dict]:
        """去重：保留置信度更高的框（1的IOU阈值更宽松）"""
        unique_detections = []
        
        for det in detections:
            is_duplicate = False
            current_char = det['char']
            current_box = det['box']
            current_conf = det['conf']
            if(current_char==".")or(current_char=="-"):
                unique_detections.append(det)
                continue
            threshold = self.iou_threshold_1 if current_char == '1' else self.iou_threshold_normal
            
            # 检查重复并保留高置信度
            for idx, unique_det in enumerate(unique_detections):
                if unique_det['char'] == current_char:
                    iou = self.calculate_iou_xaxis(current_box, unique_det['box'])
                    if iou > threshold:
                        is_duplicate = True
                        if current_conf > unique_det['conf']:
                            unique_detections[idx] = det
                        break
            
            if not is_duplicate:
                unique_detections.append(det)
        
        return unique_detections

    def sort_and_combine(self, detections: List[Dict]) -> str:
        """排序并组合成字符串，新增规则：无小数点且首字符是0则加小数点"""
        sorted_detections = sorted(detections, key=lambda d: (d['box'][0] + d['box'][2])/2)
        raw_result = ''.join([d['char'] for d in sorted_detections])
        
        # 新增规则处理
        if '.' not in raw_result and len(raw_result) > 0 and raw_result[0] == '0':
            if len(raw_result) == 1:
                final_result = raw_result
            else:
                final_result = raw_result[0] + '.' + raw_result[1:]
        else:
            final_result = raw_result
        
        return final_result

    def convert_str_to_number(self, num_str: str) -> Tuple[bool, Union[int, float, None]]:
        """
        容错转换字符串到数字（最终对外暴露的转换方法）
        :param num_str: 组合后的数字字符串
        :return: (是否成功, 数字/None)
        """
        if not num_str:
            return False, None
        
        # 容错清理
        cleaned_str = num_str.strip()
        # 处理多个小数点
        if cleaned_str.count('.') > 1:
            first_dot_idx = cleaned_str.index('.')
            cleaned_str = cleaned_str[:first_dot_idx+1] + cleaned_str[first_dot_idx+1:].replace('.', '')
        # 处理错位负号
        if '-' in cleaned_str and cleaned_str.index('-') != 0:
            cleaned_str = cleaned_str.replace('-', '')
        # 空字符串/仅符号
        if cleaned_str in ['', '-', '.', '-.']:
            return False, None
        
        # 尝试转换
        try:
            num = float(cleaned_str)
            if(num>=10000):
                num=num*0.01
            if num.is_integer():
                num = int(num)
            return True, num
        except ValueError:
            return False, None

    def detect_single_image(self, image: Union[str, np.ndarray]) -> Tuple[bool, Union[int, float, None]]:
        """
        核心方法：输入单张图片 → 返回(是否成功, 数字/None)
        :param image: 图片路径 或 OpenCV格式的np.ndarray
        :return: (success: bool, result: int/float/None)
        """
        # 1. 加载图片（兼容路径/数组）
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                print(f"错误：无法加载图片 {image}")
                return False, None
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            print("错误：图片输入格式必须是路径字符串或np.ndarray")
            return False, None
        
        image_height, image_width = img.shape[:2]
        
        # 2. YOLO11推理（检测数字）
        try:
            results = self.model(img, verbose=False)  # verbose=False关闭推理日志
        except Exception as e:
            print(f"推理失败：{str(e)}")
            return False, None
        
        # 3. 解析检测结果
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                # 解析框坐标（xyxy格式）、置信度、类别ID
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 转numpy并从GPU移到CPU
                conf = box.conf[0].cpu().numpy()
                if(conf<0.2):
                    continue
                cls_id = int(box.cls[0].cpu().numpy())
                
                # 映射类别ID到字符
                if cls_id not in self.class_id_to_char:
                    continue  # 跳过未知类别
                char = self.class_id_to_char[cls_id]
                if(char=="-")and(conf<0.5):
                    continue
                    
                # 存储检测结果
                detections.append({
                    'char': char,
                    'box': (x1, y1, x2, y2),
                    'conf': conf
                })
        
        # 4. 无检测结果直接返回失败
        if not detections:
            print("未检测到任何数字/符号")
            return False, None
        
        # 5. 数字处理流水线
        # 5.1 过滤小数字框
        filtered = self.filter_small_digits(detections, image_height)
        if not filtered:
            print("所有检测框高度均不符合要求")
            return False, None
        
        # 5.2 去重（保留高置信度）
        deduplicated = self.remove_duplicate_digits(filtered)
        
        # 5.3 处理特殊字符（负号/小数点）
        processed = self.process_special_chars(deduplicated, image_height)
        
        # 5.4 排序并组合字符串
        combined_str = self.sort_and_combine(processed)
        if not combined_str:
            print("组合后无有效字符串")
            return False, None
        
        # 6. 转换为数字
        success, final_num = self.convert_str_to_number(combined_str)
        
        # 7. 返回结果
       
        
        return success, final_num
