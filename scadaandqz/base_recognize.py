import cv2
from typing import Dict, Any, Optional, Callable
import threading
import time
import logging
import traceback
from pathlib import Path
from udp import UDPSender

# ======================== 日志配置（含识别器类名）========================
def setup_logger(name: str = "RecognitionFramework") -> logging.Logger:
    """配置通用日志：分级存储+精准定位+识别器类名"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # 日志格式：时间 | 级别 | 线程 | 识别器 | 文件:行号 | 消息
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(threadName)s | 【%(name)s】 | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台输出（INFO及以上）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 全量日志文件（按识别器分类）
    file_handler = logging.FileHandler(
        log_dir / f"{name}.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 错误日志文件（按识别器分类）
    error_handler = logging.FileHandler(
        log_dir / f"{name}_error.log",
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    return logger

# ======================== 通用识别管理器 =========================
class BaseRecognitionManager:
    def __init__(self,recognizer_class: Callable):
        """
        通用识别管理器
        :param recognizer_class: 识别器类（如OCRRecognizer），需实现 __init__(**model_paths) 和 recognize(frame, params)
        :param push_func: 自定义推送函数，默认使用HTTP POST
        """
        # 识别器标识（核心：记录类名用于日志）
        self.recognizer_class = recognizer_class
        self.recognizer_class_name = recognizer_class.__name__
        
        # 日志初始化（含识别器类名）
        self.logger = setup_logger(f"RecognitionManager_{self.recognizer_class_name}")
        
        # 指令控制
        self.cmd_queue = []
        self.cmd_lock = threading.Lock()
        
        # 工作线程
        self.worker_thread: Optional[threading.Thread] = None
        self.worker_running = False
        
        # 模型状态
        self.recognizer: Optional[Any] = None
        self.model_init_done = False
        self.model_init_error = ""
        self.model_paths: Dict[str, str] = {}
        
        # 识别状态
        self.current_camera_id = -1
        self.is_recognizing = False
        self.camera_handles = {}  # 摄像头句柄缓存
        
        # 推送配置
        self.udp_sender: Optional[UDPSender] = None
        self.udp_target_ip =""
        self.udp_target_port =-1
        self.picdir=""
        self.result_lock = threading.Lock()
        self.latest_recognition_result = {
             "jxt_yc": [],
            "jxt_yx": [],
            "qz_yc": [],
            "qz_yx": []
        }  # 本地最新结果缓存
        # 重连配置
        self.reconnect_count = 0
        self.MAX_RECONNECT_COUNT = 0  # 0=无限重连
        self.force_update_flag = False
    def _init_model(self):
        """初始化识别模型（工作线程内执行）"""
        if not self.model_init_done and self.model_paths:
            try:
                self.udp_target_ip = self.model_paths["UDPSender"]["target_ip"]
                self.udp_target_port = self.model_paths["UDPSender"]["target_port"]
                self.picdir=self.model_paths["save_image_path"]
                self.udp_sender=UDPSender(self.udp_target_ip,self.udp_target_port)
                self.logger.info(f"【{self.recognizer_class_name}】开始初始化识别模型...")
                self.recognizer = self.recognizer_class(self.model_paths)
                self.model_init_done = True
                self.model_init_error = ""
                self.logger.info(f"【{self.recognizer_class_name}】识别模型初始化完成！")
            except Exception as e:
                self.model_init_done = False
                self.model_init_error = str(e)
                self.logger.error(f"【{self.recognizer_class_name}】模型初始化失败：{str(e)}\n{traceback.format_exc()}")

    def _get_camera_handle(self, camera_id) -> Optional[cv2.VideoCapture]:
        """获取摄像头句柄（复用缓存）"""
        if camera_id in self.camera_handles:
            cap = self.camera_handles[camera_id]
            if cap.isOpened():
                #self.logger.info(f"【{self.recognizer_class_name}】摄像头{camera_id}已打开，复用句柄")
                return cap
            else:
                cap.release()
                del self.camera_handles[camera_id]
                self.logger.warning(f"【{self.recognizer_class_name}】摄像头{camera_id}句柄失效，已释放")
        
        # 重新打开摄像头
        try:
            self.logger.info(f"【{self.recognizer_class_name}】尝试打开摄像头{camera_id}")
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # 避免缓存大量帧
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            if cap.isOpened():
                self.camera_handles[camera_id] = cap
                self.logger.info(f"【{self.recognizer_class_name}】摄像头{camera_id}打开成功，缓存句柄")
                return cap
            else:
                self.logger.error(f"【{self.recognizer_class_name}】摄像头{camera_id}打开失败")
                return None
        except Exception as e:
            self.logger.error(f"【{self.recognizer_class_name}】打开摄像头{camera_id}异常：{str(e)}\n{traceback.format_exc()}")
            return None

    def _release_camera_handle(self, camera_id: int):
        """释放指定摄像头句柄"""
        if camera_id in self.camera_handles:
            cap = self.camera_handles[camera_id]
            cap.release()
            del self.camera_handles[camera_id]
            self.logger.info(f"【{self.recognizer_class_name}】已释放摄像头{camera_id}句柄")

    def _draw_target_point_box(self, frame, result: dict, current_params: dict):
        """
        在图像中画出待测点的检测框（仅标出待测点）
        :param frame: 原始图像（OpenCV格式）
        :param result: 识别结果，包含 Res.qz_yc / Res.qz_yx
        :param current_params: 当前参数，包含 target_point 和 ycyx
        :return: 画框后的图像
        """
        try:
            target_point = current_params.get("target_point")
            if target_point is None:
                return frame

            target_id_str = str(target_point)
            ycyx = current_params.get("ycyx", "yc")
            target_list = result.get("Res", {}).get("qz_yc" if ycyx == "yc" else "qz_yx", [])
            table_box = result["visualization"].get("table_info").get("bbox")
            cell_boxes_2d = result["visualization"].get("cell_boxes_2d")
            id_coord = result["visualization"].get("id_coord")
            value_coord = result["visualization"].get("value_coord")
            rows_to_draw = set(range(1, len(cell_boxes_2d)))
            if target_id_str is not None and cell_boxes_2d:
                matched_rows = set()
                for row_idx in range(1, len(cell_boxes_2d)):
                    try:
                        text = target_list[row_idx - 1].get("id", "")
                    except Exception:
                        continue
                    if text == target_id_str:
                        matched_rows.add(row_idx)
                        break
                rows_to_draw = matched_rows
                if not rows_to_draw:
                    print("未找到点号 '%s'，可视化将不标注任何数据行", target_id_str)
            if cell_boxes_2d:
                if value_coord:
                    val_col = value_coord[1]
                    for row_idx, row_cells in enumerate(cell_boxes_2d):
                        if row_idx not in rows_to_draw:
                            continue
                        if 0 <= val_col < len(row_cells):
                            cell = row_cells[val_col]
                            offset_x, offset_y = table_box[0], table_box[1]
                            cx1, cy1, cx2, cy2 = cell[0] + offset_x, cell[1] + offset_y, cell[2] + offset_x, cell[3] + offset_y
                            cv2.rectangle(frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 255, 0), 2)
        except Exception as e:
            self.logger.warning(f"【{self.recognizer_class_name}】画待测点框失败：{str(e)}")

        return frame

    def _worker_loop(self):
        """基础工作线程主循环（子类可重写）"""
        self.logger.info(f"【{self.recognizer_class_name}】工作线程启动")
        self._init_model()
        
        current_params = {}
        while self.worker_running:
            # 处理指令队列
            current_cmd = None
            with self.cmd_lock:
                if len(self.cmd_queue) > 0:
                    current_cmd = self.cmd_queue.pop(0)
            
            if current_cmd:
                cmd_type = current_cmd[0]
                
                # 停止指令
                if cmd_type == "stop":
                    self.is_recognizing = False
                    self.current_camera_id = -1
                    self.reconnect_count = 0
                    for cam_id in list(self.camera_handles.keys()):
                        self._release_camera_handle(cam_id)
                    self.logger.info(f"【{self.recognizer_class_name}】已停止识别，释放所有摄像头句柄")
                
                # 启动/切换指令
                elif cmd_type == "start":
                    camera_id = current_cmd[1]
                    params = current_cmd[2]
                    self.force_update_flag = True
                    current_params = params
                    
                    self.is_recognizing = False
                    self.reconnect_count = 0
                    
                    if not self.model_init_done:
                        self.logger.error(f"【{self.recognizer_class_name}】模型未初始化，无法启动识别：{self.model_init_error}")
                        continue
                    
                    if camera_id == self.current_camera_id:
                        self.logger.info(f"【{self.recognizer_class_name}】摄像头ID{camera_id}未变化，复用现有配置")
                        self.is_recognizing = True
                        continue
                    
                    self.logger.info(f"【{self.recognizer_class_name}】切换摄像头：{self.current_camera_id} → {camera_id}")
                    self.current_camera_id = camera_id
                    
                    cap = self._get_camera_handle(camera_id)
                    if cap is None:
                        self.logger.error(f"【{self.recognizer_class_name}】无法获取摄像头{camera_id}句柄，启动失败")
                        continue
                    
                    self.is_recognizing = True
                    self.logger.info(f"【{self.recognizer_class_name}】开始识别摄像头{camera_id}")
            
            # 基础识别逻辑（单帧）
            if self.is_recognizing and self.model_init_done and self.current_camera_id != -1:
                cap = self._get_camera_handle(self.current_camera_id)
                if cap is None:
                    self.reconnect_count += 1
                    if self.MAX_RECONNECT_COUNT > 0 and self.reconnect_count > self.MAX_RECONNECT_COUNT:
                        self.logger.error(f"【{self.recognizer_class_name}】摄像头{self.current_camera_id}重连{self.MAX_RECONNECT_COUNT}次失败，停止识别")
                        self.is_recognizing = False
                        self.reconnect_count = 0
                        time.sleep(0.5)
                        continue
                    
                    self.logger.warning(f"【{self.recognizer_class_name}】重连摄像头{self.current_camera_id}（第{self.reconnect_count}次）...")
                    time.sleep(2)
                    continue
                
                try:
                    ret, frame = cap.read()
                    if ret:
                        self.reconnect_count = 0
                        result = self.recognizer.run(frame, current_params)
                        if(result["status"]=="fail"):
                            self.logger.error(f"【{self.recognizer_class_name}】检测失败 | 错误："+result["error"])
                        stationname=current_params["stationname"]
                        qz_yc=result["Res"]["qz_yc"]
                        qz_yx=result["Res"]["qz_yx"]
                        # 画出待测点的检测框（仅标出待测点）
                        frame = self._draw_target_point_box(frame, result, current_params)
                        self.udp_sender.send_direct(stationname,current_params,self.picdir,frame,qz_yc=qz_yc,qz_yx=qz_yx,force_update_flag=self.force_update_flag)
                        self.force_update_flag = False
                        if qz_yc or qz_yx:  # 简化判断，效果相同
                            if(self.latest_recognition_result["qz_yc"]!=qz_yc)or (self.latest_recognition_result["qz_yx"]!=qz_yx):
                                with self.result_lock:  # 加锁保证修改原子性
                                    self.latest_recognition_result["qz_yc"] = qz_yc
                                    self.latest_recognition_result["qz_yx"] = qz_yx
                    else:
                        self.logger.warning(f"【{self.recognizer_class_name}】摄像头{self.current_camera_id}读取画面失败，准备重连")
                        self._release_camera_handle(self.current_camera_id)
                        time.sleep(0.5)
                except Exception as e:
                    self.logger.error(f"【{self.recognizer_class_name}】识别异常 | 摄像头{self.current_camera_id} | 错误：{str(e)}\n{traceback.format_exc()}")
                    time.sleep(0.5)
            else:
                time.sleep(0.5)
        
        # 退出清理
        for cam_id in list(self.camera_handles.keys()):
            self._release_camera_handle(cam_id)
        self.logger.info(f"【{self.recognizer_class_name}】工作线程退出，已清理所有资源")

    def start_worker(self, model_paths: Dict[str, str]):
        """启动工作线程（仅调用一次）"""
        if not self.worker_running:
            self.model_paths = model_paths
            self.worker_running = True
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                name="RecognitionWorker",
                daemon=True
            )
            self.worker_thread.start()
            self.logger.info(f"【{self.recognizer_class_name}】工作线程已启动")

    def stop_worker(self):
        """停止工作线程"""
        self.worker_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.logger.info(f"【{self.recognizer_class_name}】工作线程已停止")

    def send_start_cmd(self, camera_id,params: Dict[str, str]):
        """发送启动/切换指令"""
        with self.cmd_lock:
            self.cmd_queue = [("start", camera_id,params)]
        self.logger.info(f"【{self.recognizer_class_name}】发送启动指令 | 摄像头：{camera_id} | 参数：{params}")
        return True

    def send_stop_cmd(self):
        """发送停止指令"""
        with self.cmd_lock:
            self.cmd_queue = [("stop",)]
        self.logger.info(f"【{self.recognizer_class_name}】发送停止指令")
        return True
    
# ======================== 多帧通用识别管理器 =========================
class BatchFrameRecognitionManager(BaseRecognitionManager):
    """
    批量帧识别管理器（继承基类）
    特性：每隔指定时间采集1帧，累计N帧后批量检测
    """
    def __init__(self, recognizer_class, buffer_size=8, frame_interval=0.05):
        # 继承父类所有属性
        super().__init__(recognizer_class)
        # 批量帧配置
        self.buffer_size = buffer_size      # 累计检测帧数
        self.frame_interval = frame_interval  # 帧采集间隔（秒）
        self.force_update_flag = False

    def _worker_loop(self):
        """重写工作线程：批量帧采集+检测"""
        self.logger.info(f"【{self.recognizer_class_name}】批量帧工作线程启动 | 采集间隔：{self.frame_interval}s | 批量帧数：{self.buffer_size}")
        self._init_model()
        
        current_params = {}
        frame_buffer = []  # 帧缓存
        last_frame_time = 0  # 上一次采集时间
        
        while self.worker_running:
            # 1. 处理指令队列（复用父类逻辑）
            current_cmd = None
            with self.cmd_lock:
                if len(self.cmd_queue) > 0:
                    current_cmd = self.cmd_queue.pop(0)
            
            if current_cmd:
                cmd_type = current_cmd[0]
                
                # 停止指令（新增清空缓存）
                if cmd_type == "stop":
                    self.is_recognizing = False
                    self.current_camera_id = -1
                    self.reconnect_count = 0
                    frame_buffer.clear()
                    last_frame_time = 0
                    for cam_id in list(self.camera_handles.keys()):
                        self._release_camera_handle(cam_id)
                    self.logger.info(f"【{self.recognizer_class_name}】已停止识别，释放摄像头句柄并清空帧缓存")
                
                # 启动/切换指令
                elif cmd_type == "start":
                    camera_id = current_cmd[1]
                    params = current_cmd[2]
                    current_params = params
                    self.force_update_flag = True
                    self.is_recognizing = False
                    self.reconnect_count = 0
                    frame_buffer.clear()
                    last_frame_time = 0
                    
                    if not self.model_init_done:
                        self.logger.error(f"【{self.recognizer_class_name}】模型未初始化，无法启动识别：{self.model_init_error}")
                        continue
                    
                    if camera_id == self.current_camera_id:
                        self.logger.info(f"【{self.recognizer_class_name}】摄像头ID{camera_id}未变化，复用配置")
                        self.is_recognizing = True
                        continue
                    
                    self.logger.info(f"【{self.recognizer_class_name}】切换摄像头：{self.current_camera_id} → {camera_id}")
                    self.current_camera_id = camera_id
                    
                    cap = self._get_camera_handle(camera_id)
                    if cap is None:
                        self.logger.error(f"【{self.recognizer_class_name}】无法获取摄像头{camera_id}句柄，启动失败")
                        continue
                    
                    self.is_recognizing = True
                    self.logger.info(f"【{self.recognizer_class_name}】开始识别摄像头{camera_id} | 每隔{self.frame_interval}s采集1帧，累计{self.buffer_size}帧检测")
            
            # 2. 批量帧采集+检测逻辑
            if self.is_recognizing and self.model_init_done and self.current_camera_id != -1:
                cap = self._get_camera_handle(self.current_camera_id)
                if cap is None:
                    self.reconnect_count += 1
                    if self.MAX_RECONNECT_COUNT > 0 and self.reconnect_count > self.MAX_RECONNECT_COUNT:
                        self.logger.error(f"【{self.recognizer_class_name}】摄像头{self.current_camera_id}重连{self.MAX_RECONNECT_COUNT}次失败，停止识别")
                        self.is_recognizing = False
                        self.reconnect_count = 0
                        time.sleep(0.5)
                        continue
                    
                    self.logger.warning(f"【{self.recognizer_class_name}】重连摄像头{self.current_camera_id}（第{self.reconnect_count}次）...")
                    time.sleep(2)
                    continue
                
                # 按间隔采集帧
                try:
                    current_time = time.time()
                    if current_time - last_frame_time >= self.frame_interval:
                        ret, frame = cap.read()
                        last_frame_time=current_time
                        if ret:
                            self.reconnect_count = 0
                            frame_buffer.append(frame.copy())  # 深拷贝避免覆盖
                            self.logger.debug(f"【{self.recognizer_class_name}】采集到帧 | 缓存进度：{len(frame_buffer)}/{self.buffer_size}")
                            
                            # 缓存满则批量检测
                            if len(frame_buffer) >= self.buffer_size:
                                t_batch_start = time.time()
                                self.logger.info(f"【{self.recognizer_class_name}】帧缓存满{self.buffer_size}帧，执行批量检测")
                                try:
                                    t_start = time.time()
                                    # 优先调用批量识别方法
                                    result,image= self.recognizer.run(frame_buffer, current_params)
                                    t_infer = time.time() - t_start
                                    self.logger.info(f"【{self.recognizer_class_name}】算法推理耗时：{t_infer:.3f}s")
                                    stationname=current_params["stationname"]
                                    if(result["status"]=="fail"):
                                        self.logger.error(f"【{self.recognizer_class_name}】检测失败 | 错误："+result["error"])
                                    jxt_yc=result["Res"]["jxt_yc"]
                                    jxt_yx=result["Res"]["jxt_yx"]
                                    roi = current_params.get("roi",[])
                                    try:
                                        if(roi!=[]):
                                            x1,y1,x2,y2=roi
                                            cv2.rectangle(image, (x1-10, y1-10), (x2+10, y2+10), (255, 255, 255), 2)
                                    except Exception as e:
                                        continue
                                    t0 = time.time()
                                    self.udp_sender.send_direct(stationname,current_params,self.picdir,image,jxt_yx=jxt_yx,jxt_yc=jxt_yc,force_update_flag=self.force_update_flag)
                                    self.logger.info(f"【{self.recognizer_class_name}】UDP发送指令耗时：{(time.time()-t0)*1000:.1f}ms")
                                    self.force_update_flag = False
                                    if jxt_yc or jxt_yx:  # 简化判断，效果相同
                                        if(self.latest_recognition_result["jxt_yc"]!=jxt_yc)or (self.latest_recognition_result["jxt_yx"]!=jxt_yx):
                                            with self.result_lock:  # 加锁保证修改原子性
                                                self.latest_recognition_result["jxt_yc"] = jxt_yc
                                                self.latest_recognition_result["jxt_yx"] = jxt_yx
                                                
                                    t_batch_total = time.time() - t_batch_start
                                    self.logger.info(f"【{self.recognizer_class_name}】单批次总耗时（含采集结束到清空缓存）: {t_batch_total:.3f}s")
                                    frame_buffer.clear()  # 清空缓存
                                except Exception as e:
                                    self.logger.error(f"【{self.recognizer_class_name}】批量检测失败 | 错误：{str(e)}\n{traceback.format_exc()}")
                                    frame_buffer.clear()
                        else:
                            self.logger.warning(f"【{self.recognizer_class_name}】摄像头{self.current_camera_id}读取失败，准备重连")
                            self._release_camera_handle(self.current_camera_id)
                            frame_buffer.clear()
                            last_frame_time = 0
                            time.sleep(0.5)
                    else:
                        time.sleep(0.01)  # 减少CPU占用
                except Exception as e:
                    self.logger.error(f"【{self.recognizer_class_name}】帧采集异常 | 错误：{str(e)}\n{traceback.format_exc()}")
                    frame_buffer.clear()
                    last_frame_time = 0
                    time.sleep(0.5)
            else:
                frame_buffer.clear()
                last_frame_time = 0
                time.sleep(0.5)
        
        # 退出清理
        frame_buffer.clear()
        for cam_id in list(self.camera_handles.keys()):
            self._release_camera_handle(cam_id)
        self.logger.info(f"【{self.recognizer_class_name}】批量帧工作线程退出，已清理所有资源")


 
