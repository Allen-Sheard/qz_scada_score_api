# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec 文件：将 api_server.py 打包为 Windows 可执行程序。

使用方式：
    1. 确保在 Windows 上安装了 Python 和项目依赖
    2. 将本文件放到项目根目录 /workspace 下（与 api_server.py 同级）
    3. 运行：pyinstaller api_server.spec
    4. 打包产物在 dist/api_server/ 目录下

注意：
    - scadaandqz/models/ 模型权重文件已通过 datas 自动打包进 dist/api_server/
    - 如果依赖有变化，可能需要更新 hiddenimports 列表
"""

import os
from pathlib import Path

# 项目根目录
project_root = Path(SPECPATH).resolve()

# 需要打包的数据文件和目录
# (源路径, 目标路径)
datas = [
    (str(project_root / "api"), "api"),
    (str(project_root / "common"), "common"),
    (str(project_root / "scadaandqz"), "scadaandqz"),
    (str(project_root / "detect_qz.py"), "."),
    (str(project_root / "detect_scada.py"), "."),
]

# 如果存在模型配置文件，也加入
config_path = project_root / "scadaandqz" / "config.json"
if config_path.exists():
    datas.append((str(config_path), "scadaandqz"))

# 模型权重文件必须打包进去（config.json 中引用的 .pt / Paddle 推理模型）
models_dir = project_root / "scadaandqz" / "models"
if models_dir.exists():
    datas.append((str(models_dir), "scadaandqz/models"))

# 常见中文字体，用于画图显示中文
font_candidates = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
for fp in font_candidates:
    if os.path.exists(fp):
        datas.append((fp, "fonts"))

# 隐藏导入：PyInstaller 可能无法自动分析到的模块
hiddenimports = [
    # Flask
    "flask",
    "werkzeug",
    "jinja2",
    "markupsafe",
    "itsdangerous",
    "click",
    # 图像处理
    "cv2",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "numpy",
    # 项目包
    "api",
    "api.interface_qz",
    "api.interface_scada",
    "api.interface_score",
    "common",
    "common.func",
    "common.config",
    "common.logger",
    "common.global_data_index",
    "common.global_data_index_sgz",
    "common.analyse_g2",
    "common.analyse_test",
    "common.scada_algorithm",
    "scadaandqz",
    "scadaandqz.table",
    "scadaandqz.qz",
    "scadaandqz.scada",
    "scadaandqz.udp",
    "scadaandqz.base_recognize",
    # 深度学习 / OCR（PyInstaller 容易漏分析的动态导入）
    "torch",
    "torchvision",
    "paddle",
    "paddleocr",
    "ultralytics",
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "matplotlib.pyplot",
    "shapely",
    "pyclipper",
    "lmdb",
    "protobuf",
    "yaml",
    "tqdm",
    "requests",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
]

# 尝试自动发现 scadaandqz 和 common 下的所有子模块
try:
    import pkgutil
    sys.path.insert(0, str(project_root))
    for pkg_name in ["common", "scadaandqz", "api"]:
        try:
            pkg = __import__(pkg_name)
            for _, modname, ispkg in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
                if modname not in hiddenimports:
                    hiddenimports.append(modname)
        except Exception as e:
            print(f"警告：遍历包 {pkg_name} 失败: {e}")
except Exception as e:
    print(f"警告：自动发现隐藏导入失败: {e}")


a = Analysis(
    [str(project_root / "api_server.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="api_server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 保留控制台，便于查看日志
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="api_server",
)
