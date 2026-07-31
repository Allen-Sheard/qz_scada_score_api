import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Type
import cv2
from ultralytics import YOLO
from pathlib import Path
import json
from collections import defaultdict, Counter
from common.yolo11_digit_detector import YOLO11DigitDetector
from PIL import Image

def crop_roi_batch(image_list: List[np.ndarray], roi: Tuple[int, int, int, int]) -> List[np.ndarray]:
    """
    单ROI对一批图像进行裁剪（核心通用逻辑）
    :param image_list: 输入图像列表 [np.ndarray(H, W, 3), ...]
    :param roi: 单个ROI区域 (x1, y1, x2, y2)
    :return: 批量裁剪后的ROI图像列表
    """
    roi_frames = []
    x1, y1, x2, y2 = roi
    
    for frame in image_list:
        # 边界校验（防止越界，适配不同尺寸的图像）
        frame_h, frame_w = frame.shape[:2]
        roi_x1 = max(0, x1)
        roi_y1 = max(0, y1)
        roi_x2 = min(frame_w, x2)
        roi_y2 = min(frame_h, y2)
        
        # 截取ROI（空截图跳过）
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi_frame.size > 0:
            roi_frames.append(roi_frame)
    return roi_frames

def get_max_color_pixels_frame(roi_frames:List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """
    选择红色像素+绿色像素最多的整张图（修正后的核心融合规则）
    :param roi_frames: 批量ROI裁剪后的图像列表
    :return: 红+绿最多的那张ROI图像
    """
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    max_total_pixels = -1
    best_frame_idx = 0
    best_frame = None
    for idx, frame in enumerate(roi_frames):
        if frame.size == 0:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),cv2.inRange(hsv, lower_red2, upper_red2))
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        red_p = cv2.countNonZero(mask_red)
        green_p = cv2.countNonZero(mask_green)
        total_rg_p = red_p + green_p
        if total_rg_p > max_total_pixels:
            max_total_pixels = total_rg_p
            best_frame_idx = idx
            best_frame = frame.copy()
    if best_frame is None:
        raise ValueError("no frame can choose")
    return best_frame, best_frame_idx, max_total_pixels 
    
def get_brightest_image(roi_frames: List[np.ndarray]) -> np.ndarray:
    """
    选择平均亮度最高的整张图（修正后的核心融合规则）
    :param roi_frames: 批量ROI裁剪后的图像列表
    :return: 平均亮度最高的那张ROI图像
    """
    if not roi_frames:
        raise ValueError("批量ROI裁剪结果为空，无法选择亮度最高的图")
    
    # brightness_scores = []
    # for frame in roi_frames:
    #     # 方法1：RGB转灰度图计算平均亮度（更符合人眼感知）
    #     #cv2.imwrite("22.jpg",frame)
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #     avg_brightness = np.mean(gray_frame)
        
    #     # 方法2：直接计算RGB均值（可选，注释掉的备选方案）
    #     # avg_brightness = np.mean(frame)
        
    #     brightness_scores.append(avg_brightness)
    
    # 向量化计算：直接对灰度图求均值，比循环 cvtColor 更快
    brightness_scores = [
        np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if frame.size > 0 else -1
        for frame in roi_frames
    ]

    # 找到亮度最高的图像索引
    max_brightness_idx = int(np.argmax(brightness_scores))
    brightest_frame = roi_frames[max_brightness_idx]
    
    # 打印亮度信息（便于调试）
    print(f"📊 亮度评分: {[round(s, 2) for s in brightness_scores]} → 选择第{max_brightness_idx+1}张（亮度{round(brightness_scores[max_brightness_idx], 2)}）")
    
    return brightest_frame,max_brightness_idx,brightness_scores[max_brightness_idx]

def get_most_common_color_class(regions: List[np.ndarray], threshold: float = 0.1) -> int:
    """
    批量检测多个区域的颜色，统计后返回出现次数最多的类别（仅返回0或1）
    - 1: 红色主导的区域数量最多
    - 0: 绿色主导/无主导色/空区域 的数量最多（或数量相等时也返回0）
    
    参数:
        regions: 输入图像区域列表(BGR格式)
        threshold: 颜色占比阈值，超过此值认为颜色存在
        
    返回:
        int: 0 或 1（最终统计的最多类别）
    """
    # 第一步：检测每个区域，标记为1(红)或0(其他)
    count_0 = 0
    count_1 = 0
    
    # 预定义HSV颜色范围
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    for region in regions:
        # 空区域直接标记为0
        if region.size == 0:
            count_0 += 1
            continue
            
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩膜
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # 计算像素占比
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        total_pixels = region.shape[0] * region.shape[1]
        
        red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # 标记类别：1=红主导，0=其他
        if red_ratio > threshold and red_ratio > green_ratio:
            count_1 += 1
        else:
            count_0 += 1
    
    # 核心规则：红的数量多返回1，否则返回0（数量相等也返回0）
    return 1 if count_1 > count_0 else 0

def make_square_black_pad(img):
    """图像四周填充黑边成正方形"""
    h, w = img.shape[:2]
    s = max(h, w)
    t = (s-h)//2
    b = s-h-t
    l = (s-w)//2
    r = s-w-l
    out = cv2.copyMakeBorder(img, t,b,l,r, cv2.BORDER_CONSTANT, value=[0,0,0])
    return out
    

def make_img_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

class SCAlgorithmBase:
    """SCADA算法基类（统一YOLO11分类+单ROI裁剪一批图+亮度选图融合流程）"""
    def __init__(self, model_path: str,classesname:list):
        self.model_path = model_path
        self.classesname=classesname
        self.model = None
        # 初始化YOLO11分类模型
        self._load_model()
    
    def _load_model(self):
        """加载YOLO11分类模型（子类无需重写，仅模型路径不同）"""
        try:
            # 加载YOLO11分类模型（ultralytics支持分类任务）
            self.model = YOLO(self.model_path, task="classify")
            print(f"✅ {self.__class__.__name__} 模型加载完成: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"❌ {self.__class__.__name__} 模型加载失败: {str(e)}")

    def analyze(self, image_list: List[np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一分析流程（修正后：选择亮度最高的整张图送入分类）
        :param image_list: 输入图像列表 [np.ndarray(H, W, 3), ...]
        :param params: 分析参数（必须包含node_id/roi，可选conf_threshold）
        :return: 分析结果
        """
        # 1. 提取核心参数
        roi = params.get("roi")  # 单个ROI区域 (x1, y1, x2, y2)
        if not roi or len(roi) != 4:
            return {
                "status": "fail",
                "Res":"",
                "error":"roi 格式错误"
            }
        
        try:
            # 3. 单ROI对一批图像进行裁剪
            roi_frames = crop_roi_batch(image_list, roi)
            if not roi_frames:
                return {
                    "Res":"",
                    "status": "fail",
                    "error":"图像错误",
                    "max_brightness_idx":0
                }
            # 4. 选择平均亮度最高的整张图（核心修正点）
            brightest_frame ,max_brightness_idx,_= get_brightest_image(roi_frames)
            # image=image_list[max_brightness_idx]
            # x1,y1,x2,y2=roi
            
            # cv2.rectangle(image, (x1-10, y1-10), (x2+10, y2+10), (255, 255, 255), 2)
            # brightest_frame=image[y1:y2,x1:x2]
            
            new_image=make_square_black_pad(brightest_frame)
            # 给市北用的
            new_image=make_img_gray(new_image)

            # 5. YOLO11分类推理
            results = self.model(new_image)
            top_result = results[0]
            # 6. 解析分类结果
            top_class = self.classesname[top_result.probs.top1]   
            print(top_class)
            return {
                "error":"",
                "status": "success",
                "Res":top_class,
                "max_brightness_idx":max_brightness_idx
            }
        
        except Exception as e:
            print(e)
            return {
                "status": "fail",
                "Res":"",
                "error":e,
                "max_brightness_idx":0
            }
    def _get_algorithm_type(self) -> str:
        """获取算法类型标识（子类必须实现）"""
        raise NotImplementedError
    
class SCAlgorithm(SCAlgorithmBase):
    def _get_algorithm_type(self) -> str:
        return "sc"

class KGAlgorithm(SCAlgorithmBase):
    def _get_algorithm_type(self) -> str:
        return "kg"

class GZPAlgorithm(SCAlgorithmBase):
    def __init__(self, model_path: str=None, classesname: list = None):
        self.model_path = model_path
        self.classesname = classesname
        self.detector = None
        self._load_model()
    
    def _load_model(self):
        pass
        
    def analyze(self, image_list: List[np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        重写分析流程：数字识别专用逻辑
        :param image_list: 输入图像列表 [np.ndarray(H, W, 3), ...]
        :param params: 分析参数（必须包含roi）
        :return: 分析结果
        """
        # 1. 提取核心参数
        roi = params.get("roi")
        if not roi or len(roi) != 4:
            return {
                "status": "fail",
                "Res": "",
                "error": "roi 格式错误"
            }
        
        try:
            # 2. 单ROI对一批图像进行裁剪
            roi_frames = crop_roi_batch(image_list, roi)
            if not roi_frames:
                return {
                    "Res": "",
                    "status": "fail",
                    "error": "图像错误",
                    "max_brightness_idx":0
                }
            
            # 3. 选择平均亮度最高的图像

            #brightest_frame,max_brightness_idx,brightest_s = get_brightest_image(roi_frames)
            brightest_frame, max_brightness_idx, brightest_s = get_max_color_pixels_frame(roi_frames)
            results=get_most_common_color_class([brightest_frame])
            # 4. 数字检测
            return {
                "error": "",
                "status": "success",
                "Res": str(results),
                "max_brightness_idx":max_brightness_idx
            }
        except Exception as e:
            return {
                "status": "fail",
                "Res": "",
                "error": str(e),
                "max_brightness_idx":0
            }
    def _get_algorithm_type(self) -> str:
        return "gzp"
    
class GZPAlgorithm_L(SCAlgorithmBase):
    def __init__(self, model_path: str=None, classesname: list = None):
        self.model_path = model_path
        self.classesname = classesname
        self.detector = None
        self._load_model()
    
    def _load_model(self):
        pass
        
    def analyze(self, image_list: List[np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        重写分析流程：数字识别专用逻辑
        :param image_list: 输入图像列表 [np.ndarray(H, W, 3), ...]
        :param params: 分析参数（必须包含roi）
        :return: 分析结果
        """
        # 1. 提取核心参数
        roi = params.get("roi")
        if not roi or len(roi) != 4:
            return {
                "status": "fail",
                "Res": "",
                "error": "roi 格式错误"
            }
        
        try:
            # 2. 单ROI对一批图像进行裁剪
            roi_frames = crop_roi_batch(image_list, roi)
            if not roi_frames:
                return {
                    "Res": "",
                    "status": "fail",
                    "error": "图像错误",
                    "max_brightness_idx":0
                }
            
            # 3. 选择平均亮度最高的图像
            brightest_frame,max_brightness_idx,brightest_s = get_brightest_image(roi_frames)
            # lower_red1 = np.array([0, 50, 50])
            # upper_red1 = np.array([10, 255, 255])
            # lower_red2 = np.array([170, 50, 50])
            # upper_red2 = np.array([180, 255, 255])
            # hsv = cv2.cvtColor(brightest_frame, cv2.COLOR_BGR2HSV)
            
        
            # 创建颜色掩膜
            # mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            # mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            # mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            # red_pixels = cv2.countNonZero(mask_red)
            # total_pixels = brightest_frame.shape[0] * brightest_frame.shape[1]
            # red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0
            # print(f"📊 红色评分: {round(red_ratio, 2) }")
            # if(red_ratio>0.5)or(brightest_s>70):
            #     results=1
            # else:
            #     results=0
            b, g, r = cv2.split(brightest_frame)
            diff_threshold = 10  # 三通道差值阈值，可调整

            b_i = b.astype(np.int16)
            g_i = g.astype(np.int16)
            r_i = r.astype(np.int16)

            # 是否近似灰度
            is_gray = (
                (np.abs(b_i - g_i) <= diff_threshold) &
                (np.abs(g_i - r_i) <= diff_threshold) &
                (np.abs(b_i - r_i) <= diff_threshold)
            )

            # 灰度值（近似亮度）
            gray_value = ((b_i + g_i + r_i) / 3).astype(np.uint8)

            # 保留：非灰度区域 或 灰度但亮度 < 20 的区域
            mask_color = (~is_gray) | (gray_value < 20)


            # 2. 在保留区域上计算平均亮度（使用 HSV 的 V 通道）
            hsv = cv2.cvtColor(brightest_frame, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            avg_brightness = np.mean(v_channel[mask_color])
            print(f"除去灰色区域的亮度评分: {round(avg_brightness, 2) }")
            if(avg_brightness>125):
                results=1
            else:
                results=0
            return {
                "error": "",
                "status": "success",
                "Res": str(results),
                "max_brightness_idx":max_brightness_idx
            }
        except Exception as e:
            print(e)
            return {
                "status": "fail",
                "Res": "",
                "error": str(e),
                "max_brightness_idx":0
            }
    def _get_algorithm_type(self) -> str:
        return "gzp_l"
    
class DZAlgorithm(SCAlgorithmBase):
    def _get_algorithm_type(self) -> str:
        return "dz"

class JDDZAlgorithm(SCAlgorithmBase):
    def _get_algorithm_type(self) -> str:
        return "jddz"
    
class DtextAlgorithm(SCAlgorithmBase):
    """数字识别算法类（融合YOLO11DigitDetector）"""
    
    def __init__(self, model_path: str, classesname: list = None):
        # 重写初始化，加载数字检测模型
        self.model_path = model_path
        self.classesname = classesname
        self.detector = None
        self._load_model()
    
    def _load_model(self):
        """重写加载模型方法，初始化数字检测器"""
        try:
            # 初始化YOLO11数字检测器
            self.detector = YOLO11DigitDetector(
                model_path=self.model_path,
                image_height_threshold=0,
                iou_threshold_normal=0.7,
                iou_threshold_1=0.9
            )
            print(f"✅ {self.__class__.__name__} 模型加载完成: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"❌ {self.__class__.__name__} 模型加载失败: {str(e)}")
    
    def analyze(self, image_list: List[np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        重写分析流程：数字识别专用逻辑
        :param image_list: 输入图像列表 [np.ndarray(H, W, 3), ...]
        :param params: 分析参数（必须包含roi）
        :return: 分析结果
        """
        # 1. 提取核心参数
        roi = params.get("roi")
        if not roi or len(roi) != 4:
            return {
                "status": "fail",
                "Res": "",
                "error": "roi 格式错误"
            }
        
        try:
            # 2. 单ROI对一批图像进行裁剪
            roi_frames = crop_roi_batch(image_list, roi)
            if not roi_frames:
                return {
                    "Res": "",
                    "status": "fail",
                    "error": "图像错误",
                    "max_brightness_idx":0
                }
            
            # 3. 选择平均亮度最高的图像
            brightest_frame,max_brightness_idx,_= get_brightest_image(roi_frames)
            
            # 4. 数字检测
            success, final_num = self.detector.detect_single_image(brightest_frame)
            
            if success:
                return {
                    "error": "",
                    "status": "success",
                    "Res": str(final_num),
                    "max_brightness_idx":max_brightness_idx
                }
            else:
                return {
                    "status": "fail",
                    "Res": "",
                    "error": "未检测到有效数字",
                    "max_brightness_idx":0
                }
        
        except Exception as e:
            return {
                "status": "fail",
                "Res": "",
                "error": str(e),
                "max_brightness_idx":0
            }
    
    def _get_algorithm_type(self) -> str:
        return "dtext"

class AlgorithmScheduler:
    """
    算法调度器（预加载所有算法，从JSON配置文件读取scadamodel节点）
    """
    # 算法类型与类的映射表
    ALGORITHM_MAP: Dict[str, Type[SCAlgorithmBase]] = {
        "sc": SCAlgorithm,
        "kg": KGAlgorithm,
        "gzp": GZPAlgorithm,
        "dz": DZAlgorithm,
        "jddz": JDDZAlgorithm,
        "dtext":DtextAlgorithm,
        "gzp_l":GZPAlgorithm_L
    }
    
    def __init__(self, scada_model_config):
        """
        初始化调度器
        :param config_path: JSON配置文件路径（默认./config.json）
        """
        self.algorithm_instances: Dict[str, SCAlgorithmBase] = {}
        # 1. 加载JSON配置文件中的scadamodel节点
        self.algorithm_configs =scada_model_config
        # 2. 预加载所有算法实例
        self._preload_all_algorithms()
    
    def _preload_all_algorithms(self):
        """预加载所有配置的算法实例"""
        print("\n======== 开始预加载所有算法 ========")
        for algorithm_type, config in self.algorithm_configs.items():
            try:
                # 统一算法类型为小写
                algorithm_type_lower = algorithm_type.lower()
                
                # 校验算法类型是否支持
                if algorithm_type_lower not in self.ALGORITHM_MAP:
                    print(f"❌ 跳过不支持的算法类型：{algorithm_type}")
                    continue
                
                # 校验配置完整性
                model_path = config.get("model_path")
                classesname = config.get("classesname")
                # 创建算法实例并缓存
                algorithm_class = self.ALGORITHM_MAP[algorithm_type_lower]
                algorithm_instance = algorithm_class(
                    model_path=model_path,
                    classesname=classesname
                )
                self.algorithm_instances[algorithm_type_lower] = algorithm_instance
                print(f"✅ 预加载{algorithm_type}算法完成")
            
            except Exception as e:
                print(f"❌ 预加载{algorithm_type}算法失败：{str(e)}")
        print("======== 所有算法预加载完成 ========\n")
    
    def get_algorithm(self, algorithm_type: str) -> SCAlgorithmBase:
        """获取预加载的算法实例"""
        algorithm_type_lower = algorithm_type.lower()
        if algorithm_type_lower not in self.algorithm_instances:
            raise ValueError(
                f"{algorithm_type}算法未预加载！已加载的算法类型：{list(self.algorithm_instances.keys())}"
            )
        return self.algorithm_instances[algorithm_type_lower]
    
    def run(self, image_list,input_data: Dict[str, Any]) -> Dict[str, Any]:
        """统一调用接口"""
        # 1. 校验输入参数
        required_keys = ["algorithm_type", "yxyc_type","cime_ID"]
        for key in required_keys:
            if key not in input_data:
                return {
                "Res": {
                    "jxt_yc":[],
                    "jxt_yx":[]
                },
                "status": "fail",
                "error":f"缺少必填参数：{key}"
                },None   
        # 2. 提取参数
        algorithm_type = input_data["algorithm_type"]
        yxyc_type= input_data["yxyc_type"]
        cime_ID= input_data["cime_ID"]
        # 3. 获取预加载的算法实例
        try:
            algorithm = self.get_algorithm(algorithm_type)
        except ValueError as e:
            return {
                "Res": {
                    "jxt_yc":[],
                    "jxt_yx":[]
                },
                "status": "fail",
                "error":str(e)
                },None
        # 4. 执行分析
        result = algorithm.analyze(image_list, input_data)
        max_brightness_idx=result["max_brightness_idx"]
        if(result["status"]=="fail"):
            return {
                "Res": {
                    "jxt_yc":[],
                    "jxt_yx":[]
                },
                "status": "fail",
                "error":result["error"]
                },image_list[max_brightness_idx]
        if(yxyc_type=="yx"):
            return {
                "Res": {
                    "jxt_yc":[],
                    "jxt_yx":[{"id":cime_ID,
                               "value":result["Res"]}]
                },
                "status": "success",
                "error":""
                },image_list[max_brightness_idx]
        else:
            return {
                "Res": {
                    "jxt_yc":[{"id":cime_ID,
                               "value":result["Res"]}],
                    "jxt_yx":[]
                },
                "status": "success",
                "error":""
                },image_list[max_brightness_idx]
        

def load_json_file(json_path: str) -> dict:
    import os
    """读取JSON文件，返回字典"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在：{json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def main():
    """
    单文件测试主函数：直接指定3个路径即可测试
    """
    # ===================== 手动指定测试路径（修改这里即可） =====================
    ALG_CONFIG_PATH = "./config.json"  # 算法配置JSON路径
    TEST_IMAGE_PATH = "/workspace/jpg/scada/20260410_083650_210167.jpg"         # 测试图片路径
    INPUT_PARAMS_PATH = "/workspace/jpg/scada/20260410_083650_210167.json"    # 输入参数JSON路径
    
    try:
        # 1. 加载算法配置（读取scadamodel节点）
        alg_config = load_json_file(ALG_CONFIG_PATH)
        scada_model_config = alg_config.get("scada_model_paths", {})
        if not scada_model_config:
            raise ValueError("算法配置文件中未找到scadamodel节点")
        
        # 2. 初始化算法调度器
        scheduler = AlgorithmScheduler(scada_model_config)
        
        # 3. 加载测试图片和输入参数
        test_image = cv2.imread(TEST_IMAGE_PATH)
        input_params = load_json_file(INPUT_PARAMS_PATH)
        
        # 4. 执行算法测试
        print("======== 开始执行算法测试 ========")
        print(f"测试图片：{TEST_IMAGE_PATH}")
        print(f"输入参数：{json.dumps(input_params, ensure_ascii=False, indent=2)}")
        
        image_list = [test_image]  # 算法接收图片列表格式
        result, selected_image = scheduler.run(image_list, input_params)
        
        # 5. 打印测试结果
        print("\n======== 测试结果 ========")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")

if __name__ == "__main__":
    main()
