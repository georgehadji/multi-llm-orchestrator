@echo off
REM AI Orchestrator IDE - Start Script
REM ===================================
REM This script starts the IDE server with the React frontend.

cd /d "%~dp0"

echo.
echo ============================================================
echo   AI Orchestrator IDE - Startup
echo ============================================================
echo.

REM Check if frontend is built
if not exist "ide_frontend\dist\index.html" (
    echo [! ] Frontend not built. Building now...
    cd ide_frontend
    call npm run build
    if errorlevel 1 (
        echo [ERROR] Frontend build failed!
        cd ..
        exit /b 1
    )
    cd ..
)

echo [+] Starting IDE server on http://localhost:8765
echo [+] Frontend: Enabled
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Run the hardened server: loopback bind, explicit CORS allowlist (SEC-001).
REM This used to run a standalone copy of the server that bound every interface
REM with a wildcard CORS policy and no auth, despite the banner above saying
REM localhost. That module has been deleted.
python -m orchestrator.ide_backend.launch
