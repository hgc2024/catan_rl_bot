@echo off
if not exist venv (
    echo venv not found! Please run reinstall_env.bat first.
    pause
    exit /b
)
call venv\Scripts\activate
cmd /k
