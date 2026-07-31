@echo off
chcp 65001 >nul
REM Windows 一键打包脚本
REM 用法：将本文件放到项目根目录，双击运行

echo ==========================================
echo 开始打包 QZ/SCADA API 服务为 Windows exe
echo ==========================================

REM 检查 pyinstaller 是否安装
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未安装 pyinstaller，请先执行：pip install pyinstaller
    pause
    exit /b 1
)

REM 清理旧产物
echo [1/4] 清理旧产物...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM 执行打包
echo [2/4] 执行 PyInstaller 打包...
pyinstaller windows_build\api_server.spec
if errorlevel 1 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

REM 复制模型文件（根据 config.json 中的实际路径调整）
echo [3/4] 复制模型文件和配置...
if exist "scadaandqz\config.json" (
    copy /y "scadaandqz\config.json" "dist\api_server\scadaandqz\" >nul
    echo       已复制 scadaandqz\config.json
)

REM 复制中文字体（如果有）
if exist "fonts" (
    xcopy /s /e /i /y "fonts" "dist\api_server\fonts\" >nul
    echo       已复制 fonts 目录
)

REM 复制测试数据（可选，如果希望exe自带示例数据）
REM if exist "test" (
REM     xcopy /s /e /i /y "test" "dist\api_server\test\" >nul
REM     echo       已复制 test 目录
REM )

echo [4/4] 打包完成！
echo.
echo 产物目录：dist\api_server\
echo 启动文件：dist\api_server\api_server.exe
echo.
pause
