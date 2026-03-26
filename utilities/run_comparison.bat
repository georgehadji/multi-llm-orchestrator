@echo off
echo ==========================================
echo AI Orchestrator Directory Comparison
echo ==========================================
echo.
echo Scanning D: drive...
python scan_d.py
echo.
echo Scanning E: drive...
python scan_e.py
echo.
echo Comparing results...
python compare_results.py
echo.
echo Done! Check directory_comparison_report.txt
echo.
pause
