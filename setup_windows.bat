@echo off
setlocal
cd /d "%~dp0"
set "PIP_NO_INDEX="
set "YOLO_CONFIG_DIR=%~dp0.ultralytics"

set "PY_CMD="
where py >nul 2>&1
if errorlevel 1 goto no_python

py -3.10 --version >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.10"
if not "%PY_CMD%"=="" goto python_found

py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.11"
if not "%PY_CMD%"=="" goto python_found

:no_python
echo [ERROR] Python 3.10 or 3.11 was not found through the py launcher.
echo Install Python 3.10 or 3.11 and run this script again.
goto fail

:python_found
echo Using %PY_CMD%
%PY_CMD% --version

if exist "%~dp0.venv\Scripts\python.exe" goto check_venv

echo Creating .venv
%PY_CMD% -m venv "%~dp0.venv"
if errorlevel 1 (
    echo [ERROR] Failed to create .venv.
    goto fail
)
goto install

:check_venv
echo Checking existing .venv
"%~dp0.venv\Scripts\python.exe" -c "import sys; print(sys.executable)" >nul 2>&1
if not errorlevel 1 goto install
echo [ERROR] Existing .venv is damaged.
echo Delete .venv manually, then run this script again.
goto fail

:install
echo Checking interpreter and pip
"%~dp0.venv\Scripts\python.exe" -c "import sys; print(sys.executable)"
if errorlevel 1 goto fail
"%~dp0.venv\Scripts\python.exe" -m pip --version
if errorlevel 1 goto fail

echo Updating packaging tools
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel -i https://pypi.org/simple --timeout 60 --retries 5
if errorlevel 1 goto fail

echo Installing PyTorch CPU packages
"%~dp0.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --timeout 60 --retries 5
if errorlevel 1 goto fail

echo Installing requirements
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt" -i https://pypi.org/simple --timeout 60 --retries 5
if errorlevel 1 goto fail

echo Testing PyTorch import
"%~dp0.venv\Scripts\python.exe" -c "import torch; print('torch:', torch.__version__); print(torch.rand(2,3)); print('cuda:', torch.cuda.is_available())"
if errorlevel 1 goto fail

echo Verifying installation
"%~dp0.venv\Scripts\python.exe" "%~dp0verify_install.py"
if errorlevel 1 (
    echo [ERROR] Installation verification failed.
    goto fail
)

echo Setup complete.
pause
exit /b 0

:fail
echo Setup failed.
pause
exit /b 1
