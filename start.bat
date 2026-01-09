@echo off
chcp 65001 >nul
echo ========================================
echo   Response to Chat API Proxy
echo ========================================
echo.

cd /d "%~dp0"

REM 检查是否存在虚拟环境
if exist "venv\Scripts\activate.bat" (
    echo [INFO] 激活虚拟环境...
    call venv\Scripts\activate.bat
)

REM 检查依赖是否安装
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [INFO] 首次运行，正在安装依赖...
    pip install -r requirements.txt
    echo.
)

echo [INFO] 启动服务...
echo [INFO] 服务地址: http://localhost:8000
echo [INFO] API 文档: http://localhost:8000/docs
echo [INFO] 按 Ctrl+C 停止服务
echo.

python main.py

pause
