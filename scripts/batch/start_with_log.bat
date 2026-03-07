@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo Starting dashboard server...
echo Logs will be saved to dashboard.log
python start_dashboard.py > dashboard.log 2>&1
