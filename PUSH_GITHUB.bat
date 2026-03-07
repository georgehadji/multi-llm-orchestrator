@echo off
REM Push AI Orchestrator updates to GitHub

cd /d "D:\Vibe-Coding\Ai Orchestrator"

echo ========================================
echo Push to GitHub
echo ========================================
echo.

REM Check if there are changes
git status --short
if %errorlevel% neq 0 (
    echo Error: Not a git repository or git not found
    pause
    exit /b 1
)

echo.
echo Press any key to push changes...
pause > nul

REM Add all changes
git add -A

REM Commit with message
git commit -m "feat: Add issue tracking, Slack and Git integrations

New features:
- Issue tracking integration (Jira/Linear) with RICE scoring
- Slack notifications and slash commands  
- Git service (GitHub/GitLab) for CI/CD workflows
- Sync scripts for drive management
- Documentation updates

Author: Georgios-Chrysovalantis Chatzivantsidis"

REM Push to main branch
git push origin main

echo.
echo ========================================
echo Push complete!
echo ========================================
pause
