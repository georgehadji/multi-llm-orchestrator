@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  🚀 MISSION CONTROL v6.0                                ║
echo ║  Starting dashboard...                                  ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
cd /d "%~dp0"
python start_mission_control.py
pause
