@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON=%CD%\.venv_v42\Scripts\python.exe"
if not exist "%PYTHON%" (
    py -3.12 -m venv ".venv_v42"
    if errorlevel 1 goto :SETUP_ERROR
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :SETUP_ERROR
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto :SETUP_ERROR
)
"%PYTHON%" -c "import numpy,scipy,jax,matplotlib,PIL,imageio;assert numpy.__version__.startswith('2.');assert jax.__version__=='0.9.0.1'"
if errorlevel 1 goto :SETUP_ERROR
"%PYTHON%" -u distributed_runner.py --participant A --test
if errorlevel 1 goto :PREFLIGHT_ERROR
"%PYTHON%" -u distributed_runner.py --participant A
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo ================================================
if "%EXIT_CODE%"=="0" (
    echo [OK] All 3 series meet the declared checks.
) else (
    echo [WARNING/FAILED] Some series incomplete or convergence not confirmed.
)
echo SEND THIS ARCHIVE: results_A\Send_to_AZ.zip
echo See: results_A\SUMMARY.json
echo ================================================
pause
exit /b %EXIT_CODE%
:PREFLIGHT_ERROR
echo [FAILED] CFD preflight failed. Main series not started.
pause
exit /b 32
:SETUP_ERROR
echo [FAILED] Python 3.12 or package installation error.
echo See README.md
pause
exit /b 31
