@echo off
setlocal
echo =====================================================================
echo   iPhone 16 Pro Max ^& 18 Pro Max - Smart Photo ^& Video Cleaner
echo   Mode B: Combined Folder Backup + PC Archive + USB-C Prune (Win 11)
echo =====================================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/3] Creating local Python virtual environment...
    python -m venv .venv
)

echo [2/3] Installing / verifying dependencies (this takes ~30s on first run)...
.venv\Scripts\python.exe -m pip install --upgrade pip -q
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo [3/3] Launching Dashboard at http://127.0.0.1:8090 ...
echo (If you have thousands of photos, watch the scan progress below!)
start http://127.0.0.1:8090
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8090
pause
