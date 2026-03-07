@echo off
cd /d "%~dp0"
echo ==========================================
echo  LLM Orchestrator Dashboard Server
echo ==========================================
echo.
echo Starting server on http://localhost:8888
echo.
echo Press Ctrl+C to stop
echo.
python start_dashboard.py
pause
