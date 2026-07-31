# Windows 打包说明

本文档说明如何将 Linux Docker 环境中的项目打包为 Windows 可执行程序（exe）。

---

## 重要前提

**Linux 无法直接生成 Windows exe 文件**。必须在 Windows 环境（物理机、虚拟机、Windows 容器或 CI 的 Windows Runner）中执行 PyInstaller 打包。

推荐三种快速实现方式：

| 方式 | 适用场景 | 复杂度 |
|------|---------|--------|
| **GitHub Actions 自动打包** | 没有 Windows 电脑， quickest | 低 |
| **导出到 Windows 电脑打包** | 有 Windows 电脑 | 低 |
| **Windows Docker 容器打包** | 有 Windows 宿主机 + Docker Desktop | 中 |

---

## 方式一：GitHub Actions 自动打包（推荐）

项目已配置 `.github/workflows/build-windows-exe.yml`，推送到 GitHub 后会自动在 Windows Runner 上打包。

### 步骤

1. 将项目推送到 GitHub 仓库
2. 进入 GitHub 仓库页面 → Actions → Build Windows EXE
3. 点击 Run workflow 手动触发，或等待 push 后自动触发
4. 打包完成后，在 Actions 页面下载 `api_server_windows` artifact

### 产物

下载的 zip 解压后就是 `dist/api_server/` 目录，包含：

```
api_server/
├── api_server.exe          # 启动文件
├── _internal/              # Python 运行环境和依赖库
├── api/
├── common/
├── scadaandqz/
├── detect_qz.py
└── detect_scada.py
```

---

## 方式二：导出到 Windows 电脑打包

### Linux Docker 中导出项目

在 Docker 容器内执行：

```bash
cd /workspace
bash windows_build/export_for_windows.sh
```

会生成 `project_for_windows.tar.gz`，将其下载到 Windows 电脑并解压。

### Windows 上打包

1. 安装 Python 3.10
2. 安装项目依赖：
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```
3. 在项目根目录执行：
   ```bash
   pyinstaller windows_build/api_server.spec
   ```
4. 产物在 `dist/api_server/`

### 一键脚本

解压后，在项目根目录双击：

```
windows_build\build.bat
```

---

## 方式三：Windows Docker 容器打包

如果你有一台 Windows 10/11 Pro 电脑，可以开启 Windows 容器模式：

```bash
# 切换到 Windows 容器模式（Docker Desktop 右下角菜单）
# 然后在项目根目录执行：
docker build -f windows_build/Dockerfile.windows -t qz-scada-api-builder .
docker run --rm -v %cd%\dist_output:C:\app\dist_output qz-scada-api-builder
```

产物会输出到 `.\dist_output\api_server\`。

> 注意：Windows 容器需要 Windows 宿主机，不能在 Linux 宿主机上运行。

---

## 运行打包后的 exe

进入 `dist/api_server/` 目录，双击 `api_server.exe` 或在命令行执行：

```bash
api_server.exe
```

默认监听：

```
http://127.0.0.1:5000
```

---

## 模型文件处理

打包时 PyInstaller 会把代码和 `scadaandqz/config.json` 打包进去，但**模型权重文件通常很大**，需要额外确认：

1. 打开 `dist/api_server/scadaandqz/config.json`
2. 检查其中引用的模型文件路径
3. 确保对应模型文件已存在于 `dist/api_server/` 下

如果缺失，手动复制到 `dist/api_server/` 对应位置。

---

## 调用接口

服务启动后，即可调用同步接口：

```bash
curl -X POST http://127.0.0.1:5000/api/qz/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_path": "C:\\Users\\xxx\\Desktop\\test.jpg"}'
```

详细接口说明见 `/workspace/同步接口.md`。

---

## 常见问题

### 1. GitHub Actions 打包失败

检查 `.github/workflows/build-windows-exe.yml` 中的依赖安装是否正确。如果项目有 `requirements.txt`，确保已提交到仓库。

### 2. 运行 exe 时提示缺少模型文件

手动复制模型权重文件到 `dist/api_server/` 下对应路径。

### 3. exe 体积太大

正常现象。深度学习项目依赖 PyTorch/ONNX/Paddle 等库，打包后通常有 **500MB~2GB**。

### 4. 第一次请求很慢

模型在首次请求时加载。建议在 Windows 上设置环境变量后启动：

```bash
set PRELOAD_MODELS=1
api_server.exe
```

### 5. 如何隐藏黑色控制台窗口

将 `windows_build/api_server.spec` 中的 `console=True` 改为 `console=False`，然后重新打包。
