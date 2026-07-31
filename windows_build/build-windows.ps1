#Requires -Version 5.1
# 一键在 Windows 上安装依赖并打包 api_server.exe
# 用法：以管理员或普通用户身份打开 PowerShell，进入项目根目录，执行：
#   .\windows_build\build-windows.ps1
#
# 如果你使用 Miniforge/Conda，请先创建并激活环境：
#   conda env create -f environment.yml
#   conda activate qz-scada-api
#   .\windows_build\build-windows.ps1

param(
    [string]$PythonCmd = "python"
)

$ErrorActionPreference = "Stop"

function Test-Command($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  QZ/SCADA API Windows 一键打包脚本" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 0. Conda 环境提示
if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "检测到 Conda 环境: $env:CONDA_DEFAULT_ENV" -ForegroundColor Green
} elseif (Test-Command "conda") {
    Write-Host "检测到 conda 命令。建议先创建并激活环境：" -ForegroundColor Yellow
    Write-Host "  conda env create -f environment.yml" -ForegroundColor White
    Write-Host "  conda activate qz-scada-api" -ForegroundColor White
    Write-Host "然后重新运行本脚本。`n" -ForegroundColor Yellow
}

# 1. 检查 Python
if (-Not (Test-Command $PythonCmd)) {
    Write-Host "错误：未找到 $PythonCmd，请先安装 Python 3.10" -ForegroundColor Red
    exit 1
}

$pyVersion = & $PythonCmd --version 2>&1
Write-Host "Python 版本: $pyVersion"

# 2. 检查 pip
& $PythonCmd -m pip install --upgrade pip | Out-Null

# 3. 安装依赖
Write-Host "`n[1/4] 安装项目依赖（Windows CPU 版本）..." -ForegroundColor Yellow
if (Test-Path "requirements-windows.txt") {
    & $PythonCmd -m pip install -r requirements-windows.txt
} elseif (Test-Path "requirements.txt") {
    Write-Host "未找到 requirements-windows.txt，回退到 requirements.txt" -ForegroundColor Yellow
    & $PythonCmd -m pip install -r requirements.txt
} else {
    Write-Host "错误：未找到 requirements 文件" -ForegroundColor Red
    exit 1
}
if ($LASTEXITCODE -ne 0) { exit 1 }

# 4. 安装 PyInstaller
Write-Host "`n[2/4] 安装 PyInstaller..." -ForegroundColor Yellow
& $PythonCmd -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) { exit 1 }

# 5. 清理旧产物
Write-Host "`n[3/4] 清理旧产物..." -ForegroundColor Yellow
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

# 6. 执行打包
Write-Host "`n[4/4] 执行 PyInstaller 打包..." -ForegroundColor Yellow
& $PythonCmd -m PyInstaller windows_build\api_server.spec
if ($LASTEXITCODE -ne 0) {
    Write-Host "打包失败" -ForegroundColor Red
    exit 1
}

# 7. 检查结果
if (Test-Path "dist\api_server\api_server.exe") {
    Write-Host "`n✅ 打包成功！" -ForegroundColor Green
    Write-Host "产物目录：dist\api_server\" -ForegroundColor Green
    Write-Host "启动文件：dist\api_server\api_server.exe" -ForegroundColor Green
    Write-Host "`n运行方式：" -ForegroundColor Cyan
    Write-Host "  cd dist\api_server" -ForegroundColor White
    Write-Host "  .\api_server.exe" -ForegroundColor White
} else {
    Write-Host "`n❌ 未找到 api_server.exe，打包可能失败" -ForegroundColor Red
    exit 1
}
