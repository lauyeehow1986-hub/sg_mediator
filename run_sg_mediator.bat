@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM SG-MEDIATOR Portable CLI Launcher (Zero-Admin, Zero-PowerShell, Offline)
REM ============================================================================
set "ROOT_DIR=%~dp0"

REM Prefer bundled Python 3.11 Windows Embeddable runtime; fallback to system python
if exist "%ROOT_DIR%runtime\python\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%runtime\python\python.exe"
    set "PYTHONPATH=%ROOT_DIR%src;%ROOT_DIR%runtime\python\Lib\site-packages"
) else (
    set "PYTHON_EXE=python"
    set "PYTHONPATH=%ROOT_DIR%src"
)

REM Force 100% offline execution & local user-writable temp directory
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTHONUTF8=1"
set "TMP=%ROOT_DIR%work_tmp"
set "TEMP=%ROOT_DIR%work_tmp"
if not exist "%TEMP%" mkdir "%TEMP%"

set "INPUT_DIR=%~1"
set "OUTPUT_DIR=%~2"
if "%INPUT_DIR%"=="" set "INPUT_DIR=%ROOT_DIR%sample_inbox"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=%ROOT_DIR%output_results"

REM Default to conservative 8GB RAM CPU ceiling (<4.2GB peak) unless 16gb specified
set "MEM_PROFILE=8gb"
if /I "%~3"=="16gb" set "MEM_PROFILE=16gb"
if /I "%~4"=="16gb" set "MEM_PROFILE=16gb"

if /I "%MEM_PROFILE%"=="8gb" (
    set "OMP_NUM_THREADS=1"
    set "MKL_NUM_THREADS=1"
    set "ORT_INTRA_OP_NUM_THREADS=1"
    set "SG_WORKERS=2"
    set "DUCKDB_MEM_LIMIT=1536MB"
) else (
    set "OMP_NUM_THREADS=2"
    set "MKL_NUM_THREADS=2"
    set "ORT_INTRA_OP_NUM_THREADS=2"
    set "SG_WORKERS=4"
    set "DUCKDB_MEM_LIMIT=4096MB"
)

echo [SG-MEDIATOR] Starting Portable CLI (Profile=%MEM_PROFILE%, Threads=%OMP_NUM_THREADS%, Workers=%SG_WORKERS%)
"%PYTHON_EXE%" "%ROOT_DIR%src\cli.py" --input-dir "%INPUT_DIR%" --output-dir "%OUTPUT_DIR%" --profile "%MEM_PROFILE%" %5 %6 %7 %8
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
