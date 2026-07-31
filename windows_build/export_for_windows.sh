#!/bin/bash
# 从 Linux Docker 环境中导出项目，便于在 Windows 上打包
# 用法：
#   cd /workspace
#   bash windows_build/export_for_windows.sh
# 执行后会在 /workspace 下生成 project_for_windows.tar.gz

cd "$(dirname "$0")/.." || exit 1

OUTPUT="project_for_windows.tar.gz"

echo "开始导出项目文件..."

tar -czf "$OUTPUT" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    --exclude='logs' \
    --exclude='.tmp_scada' \
    --exclude='project_for_windows.tar.gz' \
    --exclude='test' \
    api_server.py \
    api_server_async.py \
    detect_qz.py \
    detect_scada.py \
    requirements-windows.txt \
    environment.yml \
    api/ \
    common/ \
    scadaandqz/ \
    windows_build/ \
    .github/workflows/ \
    2>/dev/null

# 显示导出包大小
SIZE=$(du -h "$OUTPUT" | cut -f1)
echo "导出完成: $OUTPUT (大小: $SIZE)"
echo "请将此文件下载到 Windows 电脑并解压，然后运行 windows_build/build.bat 或执行 build-windows.ps1 打包。"
