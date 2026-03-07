@echo off
echo.
echo 🔑 Checking API Keys...
echo.
cd /d "%~dp0"
python check_api_keys.py
pause
