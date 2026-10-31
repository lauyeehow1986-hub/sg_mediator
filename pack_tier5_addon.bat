@echo off
setlocal
REM Packages optional Tier-5 LLM binaries & GGUF weights into dist\sg_mediator_tier5_llm_addon_x64.zip (< 2.0 GB).
REM Extract the resulting zip directly into an unpacked sg_mediator\ folder on a 16 GB air-gapped PC.

set "ROOT_DIR=%~dp0"
echo [SG-MEDIATOR] Building optional Tier-5 16GB LLM add-on archive...
"%ROOT_DIR%runtime\python\python.exe" "%ROOT_DIR%src\tier5_llm_arbitrator.py" --pack-addon
if errorlevel 1 exit /b 1
endlocal
