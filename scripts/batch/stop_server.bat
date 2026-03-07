@echo off
echo.
echo 🛑 Stopping any running servers on port 8888...
echo.

:: Find and kill process using port 8888
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8888') do (
    echo Found process %%a, killing...
    taskkill /F /PID %%a 2>nul
)

echo.
echo ✅ Done! You can now start the server again.
echo.
pause
