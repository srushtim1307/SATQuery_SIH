@echo off
title SatQuery AI Launcher
echo ===================================================
echo           Starting SatQuery AI Application
echo ===================================================
echo [1/2] Starting FastAPI Backend on port 8000...
start "SatQuery AI - Backend (Port 8000)" cmd /k "cd /d %~dp0 && python -m uvicorn app.main:app --port 8000 --app-dir backend --host 127.0.0.1"

echo [2/2] Starting Vite Frontend on port 5173...
start "SatQuery AI - Frontend (Port 5173)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ===================================================
echo Both servers launched in separate windows!
echo Waiting for servers to initialize...
echo Opening browser at http://localhost:5173 ...
echo ===================================================
timeout /t 3 >nul
start http://localhost:5173
