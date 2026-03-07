@echo off
REM Sync D: drive (source of truth) to E: drive
REM This ensures both locations have identical files

echo ==========================================
echo Syncing AI Orchestrator: D: -> E:
echo ==========================================
echo.

REM Main orchestrator package
xcopy /E /I /Y "D:\Vibe-Coding\Ai Orchestrator\orchestrator\*.py" "E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\"

REM Documentation
echo Syncing docs...
xcopy /E /I /Y "D:\Vibe-Coding\Ai Orchestrator\docs\*.md" "E:\Documents\Vibe-Coding\Ai Orchestrator\docs\"

REM Examples (if exists)
if exist "D:\Vibe-Coding\Ai Orchestrator\examples" (
    echo Syncing examples...
    xcopy /E /I /Y "D:\Vibe-Coding\Ai Orchestrator\examples\*" "E:\Documents\Vibe-Coding\Ai Orchestrator\examples\"
)

REM Scripts (if exists)
if exist "D:\Vibe-Coding\Ai Orchestrator\scripts" (
    echo Syncing scripts...
    xcopy /E /I /Y "D:\Vibe-Coding\Ai Orchestrator\scripts\*" "E:\Documents\Vibe-Coding\Ai Orchestrator\scripts\"
)

REM Tests
xcopy /E /I /Y "D:\Vibe-Coding\Ai Orchestrator\tests\*.py" "E:\Documents\Vibe-Coding\Ai Orchestrator\tests\"

REM Root config files
echo Syncing root config files...
copy /Y "D:\Vibe-Coding\Ai Orchestrator\pyproject.toml" "E:\Documents\Vibe-Coding\Ai Orchestrator\"
copy /Y "D:\Vibe-Coding\Ai Orchestrator\.env.example" "E:\Documents\Vibe-Coding\Ai Orchestrator\"
copy /Y "D:\Vibe-Coding\Ai Orchestrator\.gitignore" "E:\Documents\Vibe-Coding\Ai Orchestrator\"
copy /Y "D:\Vibe-Coding\Ai Orchestrator\README.md" "E:\Documents\Vibe-Coding\Ai Orchestrator\"
copy /Y "D:\Vibe-Coding\Ai Orchestrator\CONTRIBUTING.md" "E:\Documents\Vibe-Coding\Ai Orchestrator\"

echo.
echo ==========================================
echo Sync Complete!
echo ==========================================
pause
