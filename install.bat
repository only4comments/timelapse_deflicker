@echo off
echo Installing Timelapse Deflicker dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Installation failed. Make sure Python 3.11 is on your PATH.
    pause
    exit /b 1
)
echo.
echo Done. Run with: python main.py
pause
