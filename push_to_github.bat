@echo off
REM GitHub Push Script for AI Orchestrator
REM Updates remote URL and pushes to georgehadji/multi-llm-orchestrator

echo ============================================================
echo AI Orchestrator - GitHub Push Script
echo Repository: github.com/georgehadji/multi-llm-orchestrator
echo ============================================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not installed or not in PATH
    echo Please install git from https://git-scm.com/
    pause
    exit /b 1
)

echo [1/6] Checking git installation...
git --version
echo.

echo [2/6] Initializing git repository...
if not exist ".git" (
    git init
    echo Git repository initialized
) else (
    echo Git repository already exists
)
echo.

echo [3/6] Setting up remote...
git remote remove origin 2>nul
git remote add origin https://github.com/georgehadji/multi-llm-orchestrator.git
echo Remote set to: https://github.com/georgehadji/multi-llm-orchestrator.git
echo.

echo [4/6] Adding all files...
git add .
echo Files added
echo.

echo [5/6] Committing changes...
git commit -m "Initial commit: AI Orchestrator v6.0 with CI/CD and documentation"
echo Changes committed
echo.

echo [6/6] Pushing to GitHub...
echo.
echo IMPORTANT: You will be prompted for GitHub credentials
echo - Username: georgehadji
echo - Password: Use GitHub Personal Access Token
echo.
echo To create a token: https://github.com/settings/tokens
echo Required scopes: repo (full control of private repositories)
echo.
pause
git push -u origin main

echo.
echo ============================================================
echo Push complete!
echo.
echo Next steps:
echo 1. Go to: https://github.com/georgehadji/multi-llm-orchestrator
echo 2. Enable GitHub Actions in Settings
echo 3. Enable GitHub Pages for documentation
echo ============================================================
pause
