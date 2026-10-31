@echo off
setlocal
REM OPTIONAL Tier-5 Local LLM Arbitrator setup (16gb profile only).
REM Run on a workstation WITH internet; copy the folder to the offline target afterwards.
REM Uses Portable Ollama (zero admin, zero service): models live inside models_llm\ollama,
REM server binds strictly to loopback 127.0.0.1:11434 with OLLAMA_NUM_PARALLEL=1 (~2.3 GB RSS).

set "ROOT_DIR=%~dp0"
set "OLLAMA_DIR=%ROOT_DIR%runtime\ollama"
if not exist "%OLLAMA_DIR%" mkdir "%OLLAMA_DIR%"

echo [SG-MEDIATOR] Downloading Portable Ollama Windows x64 archive...
curl.exe -fL -o "%ROOT_DIR%runtime\ollama.zip" https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip
if errorlevel 1 exit /b 1

echo [SG-MEDIATOR] Extracting Portable Ollama...
tar.exe -xf "%ROOT_DIR%runtime\ollama.zip" -C "%OLLAMA_DIR%"
if errorlevel 1 exit /b 1
del /Q "%ROOT_DIR%runtime\ollama.zip" 2>nul

set "OLLAMA_MODELS=%ROOT_DIR%models_llm\ollama"
set "OLLAMA_HOST=127.0.0.1:11434"
set "OLLAMA_NUM_PARALLEL=1"
if not exist "%OLLAMA_MODELS%" mkdir "%OLLAMA_MODELS%"

echo [SG-MEDIATOR] Starting local loopback Ollama server and pulling qwen2.5:3b-instruct-q4_K_M...
start "" /B "%OLLAMA_DIR%\ollama.exe" serve
timeout /t 5 /nobreak >nul
"%OLLAMA_DIR%\ollama.exe" pull qwen2.5:3b-instruct-q4_K_M
if errorlevel 1 exit /b 1

echo.
echo [SG-MEDIATOR] Tier-5 model ready! Before running run_sg_mediator.bat in 16gb mode on the offline PC, start:
echo   set OLLAMA_MODELS=%OLLAMA_MODELS% ^& set OLLAMA_HOST=127.0.0.1:11434 ^& set OLLAMA_NUM_PARALLEL=1 ^& "%OLLAMA_DIR%\ollama.exe" serve
endlocal
