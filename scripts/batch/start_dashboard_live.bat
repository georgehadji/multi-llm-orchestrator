@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  🎮 MISSION CONTROL LIVE v4.0                           ║
echo ║  Starting dashboard...                                  ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
cd /d "%~dp0"
python start_dashboard.py live
pause
