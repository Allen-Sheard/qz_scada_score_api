import socket
import json
import threading
import os
import time
import numpy as np
import cv2
from typing import Dict, Optional, List
import copy
def save_dict_to_json(data_dict, file_path, ensure_ascii=False, indent=4):
    """
    将字典保存为 JSON 文件
    
    参数:
        data_dict (dict): 要保存的字典数据
        file_path (str): 保存的文件路径（如: "./data.json"）
        ensure_ascii (bool): 是否强制使用 ASCII 编码，False 可正常显示中文
        indent (int): JSON 格式化缩进，增强可读性
    返回:
        bool: 保存成功返回 True，失败返回 False
    """
    try:
        # 检查输入是否为字典
        if not isinstance(data_dict, dict):
            raise TypeError("输入的数据必须是字典（dict）类型")
        
        # 获取文件目录，若不存在则创建
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # 写入 JSON 文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(
                data_dict,
                f,
                ensure_ascii=ensure_ascii,
                indent=indent
            )
        print(f"字典已成功保存到: {file_path}")
        return True
    
    except TypeError as e:
        print(f"类型错误: {e}")
        return False
    except IOError as e:
        print(f"文件写入错误: {e}")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False
class UDPSender:
    def __init__(self, target_ip: str, target_port: int, encoding: str = "utf-8"):
        """
        UDP发送器（带mid自增+数据变化检测+图片自动保存）
        :param target_ip: UDP接收端IP
        :param target_port: UDP接收端端口
        :param encoding: 报文编码（默认utf-8）
        """
        # UDP核心配置
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(3.0)
        self.target = (target_ip, target_port)
        self.encoding = encoding
        
        # mid自增配置（线程安全）
        self.mid = 1
        self.mid_lock = threading.Lock()
        
        # 历史数据缓存（用于变化检测）
        self.last_data = {
            "jxt_yc": [],
            "jxt_yx": [],
            "qz_yc": [],
            "qz_yx": []
        }
        self.data_lock = threading.Lock()

    def _get_next_mid(self) -> int:
        """获取自增的mid（线程安全）"""
        with self.mid_lock:
            current_mid = self.mid
            self.mid += 1
            return current_mid

    def _is_data_changed(self, jxt_yc: list, jxt_yx: list, qz_yc: list, qz_yx: list) -> bool:
        """
        检测数据是否发生变化（直接比较，变化时才深拷贝，避免每次json.dumps）
        :return: 有变化返回True，无变化返回False
        """
        with self.data_lock:
            if (self.last_data["jxt_yc"] != jxt_yc or
                self.last_data["jxt_yx"] != jxt_yx or
                self.last_data["qz_yc"] != qz_yc or
                self.last_data["qz_yx"] != qz_yx):
                self.last_data["jxt_yc"] =  copy.deepcopy(jxt_yc)
                self.last_data["jxt_yx"] = copy.deepcopy(jxt_yx)
                self.last_data["qz_yc"] =  copy.deepcopy(qz_yc)
                self.last_data["qz_yx"] =   copy.deepcopy(qz_yx)
                return True
            return False

    def _save_image(self, save_dir: str,current_params:dict, image: np.array) -> str:
        """
        自动保存图片到指定目录，生成不重复的文件名
        :param save_dir: 保存目录（如"path/to/pic"）
        :param image: OpenCV格式的图片（np.array）
        :return: 完整的图片路径（如"path/to/pic/20260319_153022_123456.png"）
        """
        # 1. 创建目录（不存在则创建）
        os.makedirs(save_dir, exist_ok=True)
        
        # 2. 生成不重复的文件名（时间戳+毫秒）
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        millis = int(time.time() * 1000) % 1000000  # 毫秒级，避免同一秒重复
        filename = f"{timestamp}_{millis}.jpg"
        full_path = os.path.join(save_dir, filename)
        jsonname = f"{timestamp}_{millis}.json"
        # 3. 保存图片（OpenCV格式）
        full_path2 = os.path.join(save_dir, jsonname)
        save_dict_to_json(current_params,full_path2)
        cv2.imwrite(full_path, image)
        print(f"🖼️ 图片已保存：{full_path}")
        
        return full_path

    def build_udp_packet(
        self,
        stationname: str,
        pic: str,
        jxt_yc: list,
        jxt_yx: list,
        qz_yc: list,
        qz_yx: list
    ) -> Dict:
        """按指定格式构建UDP报文"""
        return {
            "mid": self._get_next_mid(),
            "type": 8,
            "content": {
                "stationname": stationname,
                "pic": pic,
                "jxt_yc": jxt_yc,
                "jxt_yx": jxt_yx,
                "qz_yc": qz_yc,
                "qz_yx": qz_yx
            }
        }

    def send(self, packet_data: Dict) -> bool:
        """发送UDP报文"""
        try:
            json_str = json.dumps(packet_data, ensure_ascii=False, separators=(",", ":"))
            self.sock.sendto(json_str.encode(self.encoding), self.target)
            print(f"✅ UDP发送成功 | mid={packet_data['mid']} | result={json_str} | 目标：{self.target}")
            return True
        except Exception as e:
            print(f"❌ UDP发送失败 | mid={packet_data.get('mid', '未知')} | 错误：{str(e)}")
            return False

    def send_direct(
        self,
        stationname: str,
        current_params:dict,
        pic_dir: str,  # 改为图片保存目录（原pic参数）
        image: np.array,  # OpenCV格式图片
        jxt_yc: List[Dict] = [],
        jxt_yx: List[Dict] = [],
        qz_yc: List[Dict] = [],
        qz_yx: List[Dict] = [],
        force_update_flag=False
    ) -> bool:
        """
        快捷发送：数据变化才发送 + 发送时自动保存图片
        :param stationname: 站点名（如"仙居站"）
        :param pic_dir: 图片保存目录（如"path/to/pic"）
        :param image: OpenCV格式的图片数组（np.array）
        :param jxt_yc/jxt_yx/qz_yc/qz_yx: 数据列表（变化检测用）
        :return: 发送成功返回True，无变化/失败返回False
        """
        # # 1. 检测数据是否变化
        if(force_update_flag is False):
            if not self._is_data_changed(jxt_yc, jxt_yx, qz_yc, qz_yx):
                print(f"ℹ️ 数据无变化，跳过UDP发送 | 站点：{stationname}")
                return False
            
        # 异步执行图片保存 + UDP发送，避免阻塞识别主线程
        threading.Thread(
            target=self._do_send,
            args=(stationname, current_params, pic_dir, image, jxt_yc, jxt_yx, qz_yc, qz_yx),
            daemon=True
        ).start()
        return True
        #     else:
        #         # 2. 自动保存图片，获取完整路径
        #         try:
        #             pic_full_path = self._save_image(pic_dir,current_params, image)
        #         except Exception as e:
        #             print(f"❌ 图片保存失败 | 错误：{str(e)}")
        #             return False
        # print(jxt_yc)
        # print(jxt_yx)
        # print(qz_yc)
        # print(qz_yx)
        
        # # 3. 构建报文（使用保存后的图片路径）
        # packet = self.build_udp_packet(
        #     stationname=stationname,
        #     pic=pic_full_path,  # 报文里的pic字段为实际保存路径
        #     jxt_yc=jxt_yc,
        #     jxt_yx=jxt_yx,
        #     qz_yc=qz_yc,
        #     qz_yx=qz_yx
        # )
        
        # # 4. 发送UDP报文
        # return self.send(packet)
    
    def _do_send(self, stationname, current_params, pic_dir, image, jxt_yc, jxt_yx, qz_yc, qz_yx):
        """真正执行保存和发送"""
        import time
        t0 = time.time()
        try:
            pic_full_path = self._save_image(pic_dir, current_params, image)
            print(f"[耗时] UDP图片保存: {(time.time()-t0)*1000:.1f}ms")
        except Exception as e:
            print(f"❌ 图片保存失败 | 错误：{str(e)}")
            return
        
        print(jxt_yc)
        print(jxt_yx)
        print(qz_yc)
        print(qz_yx)
        
        t0 = time.time()
        packet = self.build_udp_packet(
            stationname=stationname,
            pic=pic_full_path,
            jxt_yc=jxt_yc,
            jxt_yx=jxt_yx,
            qz_yc=qz_yc,
            qz_yx=qz_yx
        )
        self.send(packet)
        print(f"[耗时] UDP构建+发送: {(time.time()-t0)*1000:.1f}ms")

    def close(self):
        """关闭UDP Socket"""
        self.sock.close()
        print(f"🔌 UDP Socket已关闭 | 目标：{self.target}")

# ------------------------------ 测试示例 ------------------------------
# if __name__ == "__main__":
#     # 1. 初始化UDP发送器
#     udp_sender = UDPSender(target_ip="127.0.0.1", target_port=8888)
    
#     # 2. 模拟生成测试图片（640x480随机图片）
#     test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
#     # 3. 第一次发送（数据有变化，会发送+保存图片）
#     udp_sender.send_direct(
#         stationname="仙居站",
#         pic_dir="./pic_save",  # 图片保存到当前目录的pic_save文件夹
#         image=test_image,
#         jxt_yc=[{"value": "1", "id": "1"}],
#         jxt_yx=[{"value": "1", "id": "1"}],
#         qz_yc=[{"value": "1", "id": "1"}],
#         qz_yx=[{"value": "1", "id": "1"}]
#     )
    
#     # 4. 第二次发送（数据无变化，跳过）
#     udp_sender.send_direct(
#         stationname="仙居站",
#         pic_dir="./pic_save",
#         image=test_image,
#         jxt_yc=[{"value": "1", "id": "1"}],  # 和第一次相同
#         jxt_yx=[{"value": "1", "id": "1"}],
#         qz_yc=[{"value": "1", "id": "1"}],
#         qz_yx=[{"value": "1", "id": "1"}]
#     )
    
#     # 5. 第三次发送（数据变化，再次发送+保存新图片）
#     udp_sender.send_direct(
#         stationname="仙居站",
#         pic_dir="./pic_save",
#         image=test_image,
#         jxt_yc=[{"value": "2", "id": "1"}],  # 数据变化
#         jxt_yx=[{"value": "1", "id": "1"}],
#         qz_yc=[{"value": "1", "id": "1"}],
#         qz_yx=[{"value": "1", "id": "1"}]
#     )
    
#     # 6. 关闭发送器
#     udp_sender.close()