@echo off
chcp 65001 >nul
echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║      LLM Orchestrator Dashboard v6.5.22 - Launcher            ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo ✅ Python found
echo 🚀 Starting Dashboard...
echo.

REM Change to script directory
cd /d "%~dp0"

REM Run the dashboard
python start_dashboard.py %*

REM Pause if error
if errorlevel 1 (
    echo.
    echo ❌ Dashboard stopped with errors
    pause
)
