@echo off
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%~dp0.ultralytics"
set "QT_PLUGIN_PATH=%~dp0.venv\Lib\site-packages\PyQt5\Qt5\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%~dp0.venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Run setup_windows.bat first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -c "import torch; exec(compile(open('main.py', encoding='utf-8').read(), 'main.py', 'exec'))"
pause
