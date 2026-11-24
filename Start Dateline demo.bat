@echo off
setlocal
title Dateline demo - keep this window open
cd /d "%~dp0"

echo.
echo   DATELINE DEMO
echo   ------------------------------------------------------------
echo   Keep this window open during the demo. Close it to stop Dateline.
echo.

rem --- 1. Python environment -------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo   [problem] The Python environment .venv is missing.
  echo   Set it up once with the steps under "Setup" in README.md, then run this file again.
  echo.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
  echo   [problem] The Python environment in .venv does not start.
  echo   Rebuild it from a normal PowerShell window in this folder:
  echo     uv venv --python 3.12 --clear .venv
  echo     uv pip install --python .venv\Scripts\python.exe -r requirements.txt
  echo.
  pause
  exit /b 1
)
if not exist ".env" (
  echo   [note] No .env file found. The stand-in engine will answer.
  echo   To use the real engine with Claude Haiku 5.5, copy .env.example to .env,
  echo   paste your OpenRouter key as OPENROUTER_API_KEY, and set
  echo   DATELINE_ENGINE=real and DATELINE_ENGINE_MODULE=pipeline.engine.
  echo.
)

rem --- 2. Already running? Just open the browser -----------------------------
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  echo   Dateline is already running. Opening it in your browser.
  start "" http://localhost:8000
  ping -n 4 127.0.0.1 >nul
  exit /b 0
)
powershell -NoProfile -Command "try { Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction Stop | Out-Null; exit 1 } catch { exit 0 }"
if errorlevel 1 (
  echo   [problem] Another program is using port 8000, so Dateline cannot start there.
  echo   Close that program or the other Dateline window, then run this file again.
  echo.
  pause
  exit /b 1
)

rem --- 3. Build the website if it is missing or older than its source --------
powershell -NoProfile -Command "$d = 'web\dist\index.html'; if (-not (Test-Path $d)) { exit 1 }; $t = (Get-Item $d).LastWriteTime; $new = Get-ChildItem web\src, web\index.html, web\package.json, web\vite.config.ts, data\presets.json -Recurse -File | Where-Object { $_.LastWriteTime -gt $t }; if ($new) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  where npm >nul 2>nul
  if errorlevel 1 (
    if exist "web\dist\index.html" (
      echo   [note] Node.js is not installed, so the website was not rebuilt. Using the last build.
    ) else (
      echo   [problem] The website has not been built and Node.js is not installed.
      echo   Install Node.js from https://nodejs.org, then run this file again.
      pause
      exit /b 1
    )
  ) else (
    echo   Building the website. This takes a few seconds...
    pushd web
    if not exist "node_modules" call npm install --no-audit --no-fund
    call npm run build
    if errorlevel 1 (
      popd
      if not exist "web\dist\index.html" (
        echo   [problem] The website build failed. The messages above say why.
        pause
        exit /b 1
      )
      echo   [note] The website build failed, so the last build is used. The messages above say why.
      pushd web
    )
    popd
    echo.
  )
)

rem --- 4. Open the browser once the server answers ---------------------------
start "" /b powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health -TimeoutSec 1 | Out-Null; Start-Process 'http://localhost:8000'; break } catch { Start-Sleep -Milliseconds 500 } }"

rem --- 5. Run the server (stops when this window closes) ---------------------
echo   Starting Dateline at http://localhost:8000 ...
echo   Your browser opens by itself when it is ready.
echo.
".venv\Scripts\python.exe" -m uvicorn api.server:app --host 127.0.0.1 --port 8000
echo.
echo   Dateline has stopped.
pause
