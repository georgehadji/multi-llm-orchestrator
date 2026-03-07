@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  🎨 ANT DESIGN DASHBOARD v3.0                           ║
echo ║  Starting dashboard...                                  ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
cd /d "%~dp0"
python start_dashboard.py antd
pause
