@echo off
setlocal

cd /d "%~dp0"

set "PORT=8501"

where py >nul 2>nul
if %errorlevel%==0 (
    start "" "http://localhost:%PORT%"
    py -m streamlit run dashboard/app.py --server.port %PORT% --server.address 0.0.0.0
    goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
    start "" "http://localhost:%PORT%"
    python -m streamlit run dashboard/app.py --server.port %PORT% --server.address 0.0.0.0
    goto :eof
)

echo Python was not found on PATH.
echo Install Python and the dashboard requirements first:
echo    pip install -r requirements.txt
pause
