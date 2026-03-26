@echo off
REM Run orchestrator with DeepSeek and Anthropic only (no OpenAI/Google)
REM This clears invalid API keys from the environment before running

setlocal

REM Clear invalid API keys from environment
set OPENAI_API_KEY=
set GOOGLE_API_KEY=

echo Running orchestrator with DeepSeek and Anthropic models only...
echo.

REM Run the orchestrator with all arguments passed
python -m orchestrator %*

endlocal
