@echo off
echo.
echo 🔄 Restarting LLM Orchestrator Server...
echo.

:: Kill any process using port 8888
echo Stopping old server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8888') do (
    taskkill /F /PID %%a 2>nul >nul
)
timeout /t 2 /nobreak >nul

echo Starting new server...
echo.

:: Start the server
cd /d "%~dp0"
python run_mission_control_standalone.py

pause
