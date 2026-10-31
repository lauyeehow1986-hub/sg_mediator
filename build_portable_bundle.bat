@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM Run this script ON YOUR DEVELOPMENT WORKSTATION (Antigravity IDE) to
REM assemble the offline Python 3.11 Windows Embeddable runtime + wheels
REM so the target machine needs ZERO internet, ZERO Admin, and ZERO PowerShell.
REM ============================================================================
set "ROOT_DIR=%~dp0"
set "RUNTIME_PY=%ROOT_DIR%runtime\python"
set "WHEELS_DIR=%ROOT_DIR%runtime\wheels"

echo [1/4] Creating runtime directories...
if not exist "%RUNTIME_PY%" mkdir "%RUNTIME_PY%"
if not exist "%WHEELS_DIR%" mkdir "%WHEELS_DIR%"
if not exist "%ROOT_DIR%models_onnx_int8" mkdir "%ROOT_DIR%models_onnx_int8"

echo [2/4] Downloading Python 3.11.9 Windows amd64 Embeddable ZIP using curl.exe (built into Windows 10/11)...
if not exist "%ROOT_DIR%runtime\python-3.11.9-embed-amd64.zip" (
    curl.exe -L -o "%ROOT_DIR%runtime\python-3.11.9-embed-amd64.zip" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
)

echo [3/4] Extracting Embeddable Python using tar.exe (built into Windows 10/11 cmd.exe)...
tar.exe -xf "%ROOT_DIR%runtime\python-3.11.9-embed-amd64.zip" -C "%RUNTIME_PY%"

echo [4/5] Installing offline binary wheels into runtime\python\Lib\site-packages...
pip install --target "%RUNTIME_PY%\Lib\site-packages" --only-binary=:all: duckdb pyyaml pymupdf rapidocr-onnxruntime onnxruntime pydicom lxml numpy scipy rapidfuzz tokenizers psutil pyahocorasick

echo [5/5] Fetching INT8 ONNX SapBERT biomedical weights and building K=16384 VQ codebook...
call "%ROOT_DIR%fetch_models.bat"
echo Done! Zip the entire folder and copy to the target 8GB/16GB non-admin workstation.
endlocal
