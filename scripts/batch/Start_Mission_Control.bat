@echo off
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  🚀 LLM Orchestrator v6.5.20 - Project Dashboard        ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo   Features: New Project ^| Improve Codebase ^| YAML Spec
echo.

:: Kill any existing server on port 8888
echo 🔍 Checking for existing server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8888') do (
    echo 🛑 Stopping old server (PID: %%a)...
    taskkill /F /PID %%a 2>nul >nul
    timeout /t 2 /nobreak >nul
)

:: Clear Python cache
echo 🧹 Clearing Python cache...
python clear_cache.py 2>nul >nul

echo 🚀 Starting server...
echo 🌐 URL: http://localhost:8888
echo 📄 Example spec: example_project_spec.yaml
echo.

cd /d "%~dp0"
python -B run_mission_control_standalone.py

pause
