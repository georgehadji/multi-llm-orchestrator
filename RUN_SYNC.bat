@echo off
REM AI Orchestrator - Quick Sync
REM Double-click to run sync from D: to E:

echo ========================================
echo AI Orchestrator - Full Sync and Cleanup
echo ========================================
echo.
echo This will:
echo   1. Sync all files from D: to E:
echo   2. Delete temporary files
echo   3. Verify critical files
echo.
echo Press CTRL+C to cancel, or
pause

cd /d "D:\Vibe-Coding\Ai Orchestrator"
python full_sync_and_cleanup.py --apply

echo.
echo ========================================
echo Sync Complete!
echo ========================================
pause
