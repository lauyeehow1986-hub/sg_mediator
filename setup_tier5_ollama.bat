@echo off
REM OPTIONAL Tier-5 LLM setup (16gb profile). Run on a workstation WITH internet; copy the folder offline afterwards.
REM Portable Ollama (no admin, no service): models are stored inside this folder, server binds to 127.0.0.1 only.
REM NOTE: not yet tested end-to-end against a real Ollama install.
set "ROOT_DIR=%~dp0"
set "OLLAMA_DIR=%ROOT_DIR%runtime\ollama"
if not exist "%OLLAMA_DIR%" mkdir "%OLLAMA_DIR%"
curl.exe -fL -o "%ROOT_DIR%runtime\ollama.zip" https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip
tar.exe -xf "%ROOT_DIR%runtime\ollama.zip" -C "%OLLAMA_DIR%"
set "OLLAMA_MODELS=%ROOT_DIR%models_llm\ollama"
set "OLLAMA_HOST=127.0.0.1:11434"
start "" /B "%OLLAMA_DIR%\ollama.exe" serve
timeout /t 5 /nobreak >nul
"%OLLAMA_DIR%\ollama.exe" pull qwen2.5:1.5b-instruct-q4_K_M
echo Done. Start the server before running with the 16gb profile:
echo   set OLLAMA_MODELS=%OLLAMA_MODELS% ^& "%OLLAMA_DIR%\ollama.exe" serve
