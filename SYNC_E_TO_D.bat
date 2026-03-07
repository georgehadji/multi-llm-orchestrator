@echo off
REM Sync E: drive to D: drive (if E: has newer files)
REM Use with caution - D: is typically the source of truth

echo ==========================================
echo Syncing AI Orchestrator: E: -> D:
echo WARNING: This overwrites D: files!
echo ==========================================
echo.
set /p confirm="Are you sure? (yes/no): "
if not "%confirm%"=="yes" exit /b

xcopy /E /I /Y "E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\*.py" "D:\Vibe-Coding\Ai Orchestrator\orchestrator\"
xcopy /E /I /Y "E:\Documents\Vibe-Coding\Ai Orchestrator\docs\*.md" "D:\Vibe-Coding\Ai Orchestrator\docs\"

echo.
echo Sync Complete!
pause
