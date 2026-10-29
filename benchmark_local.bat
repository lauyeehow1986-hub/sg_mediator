@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM SG-MEDIATOR Local Benchmark & PHI-Scrubbed Antigravity Reprompt Exporter
REM ============================================================================
set "ROOT_DIR=%~dp0"

if exist "%ROOT_DIR%runtime\python\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%runtime\python\python.exe"
    set "PYTHONPATH=%ROOT_DIR%src;%ROOT_DIR%runtime\python\Lib\site-packages"
) else (
    set "PYTHON_EXE=python"
    set "PYTHONPATH=%ROOT_DIR%src"
)

set "INPUT_DIR=%~1"
set "GOLD_FILE=%~2"
set "BENCH_OUT=%~3"
if "%INPUT_DIR%"=="" set "INPUT_DIR=%ROOT_DIR%sample_inbox"
if "%GOLD_FILE%"=="" set "GOLD_FILE=%ROOT_DIR%sample_inbox\gold_labels.csv"
if "%BENCH_OUT%"=="" set "BENCH_OUT=%ROOT_DIR%output_benchmark"

set "HF_HUB_OFFLINE=1"
set "PYTHONUTF8=1"
set "OMP_NUM_THREADS=1"
set "ORT_INTRA_OP_NUM_THREADS=1"

echo [SG-MEDIATOR] Running Local Benchmark & Generating Antigravity Reprompt Bundle...
"%PYTHON_EXE%" "%ROOT_DIR%src\benchmark_engine.py" --input-dir "%INPUT_DIR%" --gold-file "%GOLD_FILE%" --output-dir "%BENCH_OUT%" --profile 8gb
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
