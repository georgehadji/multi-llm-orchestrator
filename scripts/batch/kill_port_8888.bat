@echo off
echo Killing processes on port 8888...

REM Find and kill process using port 8888
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8888') do (
    echo Killing PID %%a
    taskkill /F /PID %%a 2>nul
)

echo.
echo Killing Python dashboard processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *dashboard*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *mission*" 2>nul

echo.
echo Done! Port 8888 should be free.
pause
