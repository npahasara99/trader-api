@echo off
setlocal

cd /d "%~dp0"

set "PORT=8501"
set "APP_PATH=dashboard/app.py"
set "PY_CMD="

where py >nul 2>nul
if %errorlevel%==0 set "PY_CMD=py"

if not defined PY_CMD (
    where python >nul 2>nul
    if %errorlevel%==0 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo Python was not found on PATH.
    echo Install Python and the dashboard requirements first:
    echo    pip install -r requirements.txt
    pause
    goto :eof
)

%PY_CMD% -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Streamlit is not installed for %PY_CMD%.
    echo Install the dashboard requirements first:
    echo    pip install -r requirements.txt
    pause
    goto :eof
)

start "Trader Dashboard" cmd /k "%PY_CMD% -m streamlit run %APP_PATH% --server.port %PORT% --server.address 0.0.0.0"

set /a WAIT_COUNT=0
:wait_for_dashboard
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing http://localhost:%PORT% -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 goto open_browser

set /a WAIT_COUNT+=1
if %WAIT_COUNT% GEQ 20 goto open_anyway
timeout /t 1 /nobreak >nul
goto wait_for_dashboard

:open_browser
start "" "http://localhost:%PORT%"
goto :eof

:open_anyway
echo Dashboard server is still starting. Opening the browser now.
start "" "http://localhost:%PORT%"
goto :eof
