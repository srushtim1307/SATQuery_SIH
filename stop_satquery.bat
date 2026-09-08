@echo off
title Stop SatQuery AI
echo Stopping SatQuery AI servers on port 8000 and 5173...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Terminating Backend (PID %%a)...
    taskkill /f /pid %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do (
    echo Terminating Frontend (PID %%a)...
    taskkill /f /pid %%a 2>nul
)
echo SatQuery AI servers stopped successfully!
timeout /t 2 >nul
