@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON=%CD%\.venv_v42\Scripts\python.exe"
if not exist "%PYTHON%" (
    py -3.12 -m venv ".venv_v42"
    if errorlevel 1 goto SETUP_ERROR
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto SETUP_ERROR
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto SETUP_ERROR
)
"%PYTHON%" -c "import numpy,scipy,jax,matplotlib,PIL,imageio; assert numpy.__version__.startswith('2.'); assert jax.__version__=='0.9.0.1'"
if errorlevel 1 goto SETUP_ERROR
"%PYTHON%" -u campaign_cd.py --participant D --test
if errorlevel 1 goto PREFLIGHT_ERROR
"%PYTHON%" -u campaign_cd.py --participant D
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo =====================================================
if "%EXIT_CODE%"=="0" (
    echo [OK] Pilot and 60-second stage completed, provisional grid agreement found.
) else (
    echo [WARNING/FAILED] See convergence report and numerical QC.
)
echo Send: results_D\Send_to_AZ.zip
echo Report: results_D\CD_ANALYSIS\final_REPORT.md
echo =====================================================
pause
exit /b %EXIT_CODE%
:PREFLIGHT_ERROR
echo [FAILED] CFD preflight did not pass. See results_D\_preflight.
pause
exit /b 32
:SETUP_ERROR
echo [FAILED] Python 3.12 or dependencies were not installed correctly.
echo See README.md.
pause
exit /b 31
