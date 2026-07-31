import csv
from pathlib import Path
import pandas as pd
import os
import shutil
from typing import Tuple, Dict
import re
import glob

def clean_str(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "0"
    return str(val).strip()

# ================= 1. 核心工具函数层 (保留原样，方便单独调试) =================

def GetSubstation(output_dir: Path) -> Dict:
    substation_files = list(output_dir.glob("*Substation*.csv"))
    if not substation_files:
        raise FileNotFoundError("未找到Substation表")
    df = pd.read_csv(substation_files[0], usecols=["mRID", "pathName"], encoding="utf-8", dtype=str)
    df = df.dropna(subset=["pathName"])
    df = df[df["pathName"] != ""]
    return df.set_index("mRID")["pathName"].to_dict()

def GetBay(output_dir: Path) -> Dict:
    bay_files = list(output_dir.glob("*Bay*.csv"))
    if not bay_files:
        raise FileNotFoundError("未找到Bay表")
    df = pd.read_csv(bay_files[0], usecols=["mRID", "pathName"], encoding="utf-8", dtype=str)
    df = df.dropna(subset=["pathName"])
    df = df[df["pathName"] != ""]
    return df.set_index("mRID")["pathName"].to_dict()

def GetPointMap(output_dir: Path, file_pattern: str, key_col: str, val_col: str = "dot_no",chanid1:str="chanid1") -> Dict:
    """通用的点号表读取函数 (合并了原来的 GetYX 和 GetYC)"""
    files = list(output_dir.glob(f"{file_pattern}*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到{file_pattern}表")
    df = pd.read_csv(files[0], encoding="utf-8", dtype=str)
    df = df.dropna(subset=[val_col])
    if chanid1 not in df.columns:
        df[chanid1] = "0"
    else:
        # 如果存在，空值也填充为 0，保证数据完整
        df[chanid1] = df[chanid1].fillna("0")
    df = df[df[val_col] != ""]
    df["combined_val"] = df[val_col].astype(str) + "_" + df[chanid1].astype(str)
    return df.set_index(key_col)["combined_val"].to_dict()

def GetPointMap2(filepath: str, key_col:list = [],chanid1:str="通道(No)") -> Dict:
    """通用的点号表读取函数 (合并了原来的 GetYX 和 GetYC)"""
    df = pd.read_csv(filepath, encoding="utf-8", dtype=str)
    required_cols = ["中文描述","数据点号"] + key_col+[chanid1]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV中缺少必须列：{col}")
    df = df.fillna("")
    if chanid1 not in df.columns:
        # 列不存在：整列赋值为 0
        df[chanid1] = "0"
    else:
        # 列存在：空值填充为 0
        df[chanid1] = df[chanid1].replace("", "0")
    df["combined_val"] = df["数据点号"].astype(str) + "_" + df[chanid1].astype(str)
    result={}
    for _, row in df.iterrows():
        # ===================== 核心：多列拼接成 MRID =====================
        mrid_parts = [str(row[col]) for col in key_col]
        mrid = ":".join(mrid_parts)  # 用下划线拼接 key_col 所有列

        # 构建固定格式的 item
        item = {
            "名称": row["中文描述"].strip(),    # 中文描述 → 名称
            "点号": row["combined_val"].strip(),  # 数据点号 → 点号
            "站名": "",
            "间隔名": "",
            "Substation": "",
            "Bay": ""
        }
        # 存入结果字典
        result[mrid] = item
    return result


def _ProcessSingleTable(output_dir: Path, Substation_dict: Dict, Bay_dict: Dict, Point_dict: Dict, label: str) -> Dict:
    """内部通用处理逻辑 (处理 Analog 或 Discrete)"""
    source_files = list(output_dir.glob(f"{label}*.csv"))
    if not source_files:
        raise FileNotFoundError(f"未找到{label}表")

    df_source = pd.read_csv(source_files[0], usecols=["mRID", "pathName", "devName", "devID"], encoding="utf-8", dtype=str)
    
    # 预加载设备缓存
    device_cache = {}
    for dev_name in df_source["devName"].dropna().unique():
        dev_files = list(output_dir.glob(f"*{dev_name}*.csv"))
        if not dev_files: continue
        
        df_dev = pd.read_csv(dev_files[0], usecols=lambda c: c in ["mRID", "Substation", "Bay"], encoding="utf-8", dtype=str)
        if "Bay" not in df_dev.columns: df_dev["Bay"] = ""
        
        df_dev = df_dev.dropna(subset=["mRID", "Substation"])
        df_dev["Bay"] = df_dev["Bay"].fillna("")
        df_dev["站名"] = df_dev["Substation"].map(Substation_dict).fillna("")
        df_dev["间隔名"] = df_dev["Bay"].map(Bay_dict).fillna("")
        device_cache[dev_name] = df_dev.set_index("mRID").to_dict(orient="index")

    # 拼接结果
    result = {}

    for _, row in df_source.iterrows():
        mrid = row["mRID"]
        path_name = row["pathName"]
        # ✅ 强制预设：间隔名、站名、Substation、Bay 键，永远不会出现 KeyError
        item = {
            "名称": "" if pd.isna(path_name) else str(path_name).strip(),
            "devName": row["devName"],
            "点号": Point_dict.get(mrid, ""),
            "站名": "",
            "间隔名": "",
            "Substation": "",
            "Bay": ""
        }
        # 匹配到设备则更新数据，未匹配则保留默认空值
        if row["devName"] in device_cache and row["devID"] in device_cache[row["devName"]]:
            item.update(device_cache[row["devName"]][row["devID"]])
        result[mrid] = item
    return result

# ================= 2. 唯一主入口 (你只需要调用这一个函数) =================

def ProcessCIMEFolder(input_path: str) -> Tuple[Dict, Dict]:
    """
    一键处理整个文件夹
    :param input_path: 文件夹路径字符串，例如 "csv_modules"
    :return: (Analog结果字典, Discrete结果字典)
    """
    output_dir = Path(input_path)
    
    # 1. 读取基础映射
    Substation = GetSubstation(output_dir)
    Bay = GetBay(output_dir)
    
    # 2. 读取点号表 (Analog对应YC, Discrete对应YX)
    Fes_YC = GetPointMap(output_dir, "FesYC", "Analog") # 遥测点号
    Fes_YX = GetPointMap(output_dir, "FesYX", "Discrete") # 遥信点号

    # 3. 并行处理两张表
    final_Analog = _ProcessSingleTable(output_dir, Substation, Bay, Fes_YC, "Analog")
    final_Discrete = _ProcessSingleTable(output_dir, Substation, Bay, Fes_YX, "Discrete")

    return final_Analog, final_Discrete

def ProcessTEXTFolder(input_path: str) -> Tuple[Dict, Dict]:
    """
    一键处理整个文件夹
    :param input_path: 文件夹路径字符串，例如 "csv_modules"
    :return: (Analog结果字典, Discrete结果字典)
    """
    # 2. 读取点号表 (Analog对应YC, Discrete对应YX)
    Fes_YC = {}
    # 匹配 input_path 下所有 yc_*.csv 文件（兼容你之前生成的 yc_0.csv、yc_1.csv...）
    yc_csv_list = glob.glob(os.path.join(input_path, "yc_*.csv"))
    # 如果没有分文件，兼容单个 yc.csv 的情况
    if not yc_csv_list:
        yc_csv_list = [os.path.join(input_path, "yc.csv")]

    for csv_file in yc_csv_list:
        if os.path.exists(csv_file):
            # 读取单个文件的字典
            yc_dict = GetPointMap2(csv_file, ["设备类型", "量测名称"])
            # yc_dict = GetPointMap2(csv_file, ["量测名称"])
            # 合并字典（重复 key 会覆盖，如需保留可调整）
            Fes_YC.update(yc_dict)
    print(len(Fes_YC.keys()))
    # ===================== 批量读取并合并 YX 字典 =====================
    Fes_YX = {}
    # 匹配 input_path 下所有 yx_*.csv 文件
    yx_csv_list = glob.glob(os.path.join(input_path, "yx_*.csv"))
    # 兼容单个 yx.csv
    if not yx_csv_list:
        yx_csv_list = [os.path.join(input_path, "yx.csv")]

    for csv_file in yx_csv_list:
        if os.path.exists(csv_file):
            yx_dict = GetPointMap2(csv_file, ["设备类型", "量测名称"])
            # yx_dict = GetPointMap2(csv_file, ["量测名称"])
            Fes_YX.update(yx_dict)
    print(len(Fes_YX.keys()))
    return Fes_YC, Fes_YX
# ===================== 3. 全局数据索引中心（多表场景核心优化） =====================
class GlobalDataIndex:
    """
    一次性加载所有CIME解析后的表，构建全量O(1)索引，全程内存复用，零重复IO
    """
    def __init__(self, stationname: str, cime_path: str,yctxt_path: list=[],yxtxt_path: list=[], output_dir: str = "./csv_modules", mode: str = "auto"):
        self.stationname = stationname
        self.cime_path = Path(cime_path)
        if(yctxt_path!=[]):
            self.yctxt_path = yctxt_path
        else:
            self.yctxt_path = None
        if(yxtxt_path!=[]):
            self.yxtxt_path = yxtxt_path
        else:
            self.yxtxt_path = None
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.station_id = None
        
        # 核心索引字典
        self.yx_keyid_map = {}  # keyid -> {点号, 名称,other...}
        self.yc_keyid_map = {}  # keyid -> {点号, 名称,,other...}
        self.file_path_index = {}  # 文件名 -> 全路径 索引
        
        # 【新增】保存原始DataFrame，供你的jxp_NameString逻辑使用
        self.relay_df = None  # RelaySignal表完整数据
        self.discrete_df = None  # Discrete表完整数据
        self.discrete_devid_mrid_map = {}  # devID -> mRID 映射（按顺序）
        
        # 初始化全流程
        self._init_cime_parse()
        if(self.yctxt_path is  None)and(self.yxtxt_path is  None):
            self._init_station_id()
        self._init_yx_yc_index()
        self._init_relay_discrete_df()  # 【新增】加载完整表

    def decode_line(self, b):
        for enc in ("utf-8", "gbk"):
            try:
                return b.decode(enc)
            except:
                continue
        return b.decode("utf-8", errors="ignore")

    def _split_cime_line(self, line: str, skip_first: bool = False) -> list:
        """
        根据 mode 选择分隔符分割 CIME 行
        mode='auto' 时自动检测：先尝试制表符，再尝试2+空格，最后1+空格
        mode='normal' 时使用 \t| {2,}
        mode='sgz' 时使用 \t| {1,}
        skip_first=True 对应 # 数据行，去掉第一个元素
        """
        line = line.strip()

        if self.mode == "sgz":
            parts = re.split(r'\t| {1,}', line)
        elif self.mode == "normal":
            parts = re.split(r'\t| {2,}', line)
        else:  # auto
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = re.split(r' {2,}', line)
                if len(parts) <= 1:
                    parts = re.split(r' {1,}', line)

        return parts[1:] if skip_first else parts

    def clear_dir(self, dir_path):
        dir_path = Path(dir_path)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
        dir_path.mkdir(exist_ok=True)

    def _init_cime_parse(self):
        """一次性解析CIME文件，拆分所有表到CSV（仅执行1次）"""
        self.clear_dir(self.output_dir)
        current_module = None
        writer = None
        out_file = None
        try:
            if self.yctxt_path and self.yxtxt_path:
                # 处理 遥测(yc) 文件列表：遍历所有 yctxt 路径
                for idx, yc_path in enumerate(self.yctxt_path):
                    # 生成不重复的csv文件名（yc_0.csv、yc_1.csv...）
                    csv_file = os.path.join(self.output_dir, f"yc_{idx}.csv")
                    self._init_text_parse(yc_path, csv_file)
                
                # 处理 遥信(yx) 文件列表：遍历所有 yxtxt 路径
                for idx, yx_path in enumerate(self.yxtxt_path):
                    # 生成不重复的csv文件名（yx_0.csv、yx_1.csv...）
                    csv_file = os.path.join(self.output_dir, f"yx_{idx}.csv")
                    self._init_text_parse(yx_path, csv_file)
            with open(self.cime_path, "rb") as f:
                for raw in f:
                    line = self.decode_line(raw).rstrip("\n")
                    if line.startswith("<") and line.endswith(">") and "::" in line:
                        if out_file:
                            out_file.close()
                        current_module = line.strip("<>").replace("::", "_")
                        out_path = self.output_dir / f"{current_module}.csv"
                        out_file = open(out_path, "w", newline="", encoding="utf-8")
                        writer = csv.writer(out_file)
                    elif line.startswith("@") and writer:
                        headers = self._split_cime_line(line)
                        writer.writerow(headers)
                    elif line.startswith("#") and writer:
                        row = self._split_cime_line(line, skip_first=True)
                        writer.writerow(row)
        except Exception as e:
            print(f"CIME解析失败：{e}")
        finally:
            if out_file:
                out_file.close()
        print(f"CIME解析完成，所有表已保存到 {self.output_dir}")

    def _init_text_parse(self,txt_path,csv_file):
        """一次性解析CIME文件，拆分所有表到CSV（仅执行1次）"""
        current_module = None
        writer = None
        out_file = None
        try:
            with open(txt_path, "r", encoding="gbk") as f_in:
                # 按行读取文本，去除空行和首尾空白字符
                lines = [line.strip() for line in f_in if line.strip()]

            with open(csv_file, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out)
                # 逐行处理：按制表符\t拆分字段（你的TXT是制表符分隔的表格）
                for line in lines:
                    # 拆分每一行数据，生成列表
                    row = line.split("\t")
                    # 写入CSV
                    writer.writerow(row)          
        except Exception as e:
            print(f"TEXT解析失败：{e}")
        finally:
            if out_file:
                out_file.close()
        print(f"TEXT解析完成，所有表已保存到 {self.output_dir}")
    

    def _init_station_id(self):
        """一次性获取站点ID，仅执行1次"""
        substation_files = list(self.output_dir.glob("*Substation*.csv"))
        if not substation_files:
            raise FileNotFoundError("未找到Substation表，CIME解析失败")
        df = pd.read_csv(substation_files[0], usecols=["mRID", "pathName"], encoding="utf-8", dtype=str)
        match_row = df[df["pathName"].str.contains(self.stationname, na=False)]
        if match_row.empty or self.stationname=="":
            self.station_id=None
            print(f"没有找到场站默认获取所有场站{self.stationname}")
        else:
            self.station_id = match_row.iloc[0]["mRID"]
        print(f"站点ID获取完成：{self.station_id}")

    def _init_yx_yc_index(self):
        """一次性构建遥信/遥测keyid索引，O(1)匹配"""
        if(self.yctxt_path is not None)and(self.yxtxt_path is not None):
            self.yc_keyid_map,self.yx_keyid_map = ProcessTEXTFolder(self.output_dir)
        else:
            self.yc_keyid_map,self.yx_keyid_map = ProcessCIMEFolder(self.output_dir)


    def _init_relay_discrete_df(self):
        """【新增】加载完整的RelaySignal和Discrete表，保留原始逻辑"""
        # 1. 加载RelaySignal表
        relay_files = list(self.output_dir.glob("*RelaySignal*.csv"))
        if relay_files:
            self.relay_df = pd.read_csv(relay_files[0], encoding="utf-8", dtype=str)
            if "Substation" in self.relay_df.columns and self.station_id is not None:
                self.relay_df = self.relay_df[self.relay_df["Substation"] == self.station_id]
            print(f"RelaySignal表加载完成，共 {len(self.relay_df)} 条数据")

        # 2. 加载Discrete表并构建devID->mRID映射
        discrete_files = list(self.output_dir.glob("*Discrete*.csv"))
        if discrete_files:
            self.discrete_df = pd.read_csv(discrete_files[0], encoding="utf-8", dtype=str,on_bad_lines="skip")
            if "Substation" in self.discrete_df.columns and self.station_id is not None:
                self.discrete_df = self.discrete_df[self.discrete_df["Substation"] == self.station_id]
            
            # 构建devID到mRID的映射（按顺序，取第一个匹配）
            devid_col = None
            mrid_col = None
            for col in self.discrete_df.columns:
                col_lower = str(col).lower()
                if "devid" in col_lower:
                    devid_col = col
                if "mrid" in col_lower:
                    mrid_col = col
            
            if devid_col and mrid_col:
                for _, row in self.discrete_df.iterrows():
                    devid = str(row[devid_col]).strip()
                    mrid = str(row[mrid_col]).strip()
                    if devid and mrid and devid not in self.discrete_devid_mrid_map:
                        self.discrete_devid_mrid_map[devid] = mrid
            
            print(f"Discrete表加载完成，共 {len(self.discrete_df)} 条数据，devID映射 {len(self.discrete_devid_mrid_map)} 条")

    def build_file_path_index(self, root_dirs: list):
        """一次性遍历所有目录/文件，构建文件名到全路径的索引"""
        self.file_path_index = {}
        for root_dir in root_dirs:
            if not root_dir:
                continue
            # 支持列表元素（如 g_path 是文件路径列表）
            entries = root_dir if isinstance(root_dir, list) else [root_dir]
            for entry in entries:
                if not entry or not os.path.exists(entry):
                    continue
                if os.path.isdir(entry):
                    for dirpath, _, filenames in os.walk(entry):
                        for filename in filenames:
                            if filename.endswith(".g"):
                                self.file_path_index[filename.lower()] = os.path.join(dirpath, filename)
                elif os.path.isfile(entry):
                    self.file_path_index[os.path.basename(entry).lower()] = entry

            # for dirpath, _, filenames in os.walk(root_dir):
            #     for filename in filenames:
            #         self.file_path_index[filename.lower()] = os.path.join(dirpath, filename)
        print(f"文件路径索引构建完成，共 {len(self.file_path_index)} 个文件")

    # def find_file(self, filename: str, ignore_case: bool = True) -> str:
    #     """O(1)查找文件"""
    #     if ignore_case:
    #         filename = filename.lower()
    #     return self.file_path_index.get(filename, "")
    def find_file(self, filename: str, ignore_case: bool = True) -> str:
        """O(1)查找文件，支持精确匹配 + 无后缀模糊匹配"""
        search_key = filename.lower() if ignore_case else filename

        # 1. 先尝试精确匹配
        if ignore_case:
            result = self.file_path_index.get(search_key, "")
        else:
            result = self.file_path_index.get(filename, "")

        if result:
            return result

        # 2. 精确没找到 → 模糊匹配：文件名 包含 输入的字符串
        for stored_filename, path in self.file_path_index.items():
            fname = stored_filename.lower() if ignore_case else stored_filename
            if search_key in fname:
                return path

        # 3. 都找不到
        return ""