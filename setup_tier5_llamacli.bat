@echo off
setlocal
REM OPTIONAL Zero-Socket Tier-5 Local LLM Arbitrator Setup (16gb profile only).
REM Designed for strict hospital enclaves that block even loopback TCP sockets (127.0.0.1).
REM Run on an internet-connected PC, then copy runtime\llama and models_llm (or run pack_tier5_addon.bat).

set "ROOT_DIR=%~dp0"
set "LLAMA_DIR=%ROOT_DIR%runtime\llama"
set "LLM_DIR=%ROOT_DIR%models_llm"
if not exist "%LLAMA_DIR%" mkdir "%LLAMA_DIR%"
if not exist "%LLM_DIR%" mkdir "%LLM_DIR%"

if not exist "%LLAMA_DIR%\llama-cli.exe" (
    echo [SG-MEDIATOR] Querying recent official llama.cpp Windows x64 CPU releases via bundled Python...
    "%ROOT_DIR%runtime\python\python.exe" -c "import json, urllib.request, pathlib; rels = json.loads(urllib.request.urlopen('https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10', timeout=30).read().decode('utf-8')); urls = [a['browser_download_url'] for r in rels for a in r.get('assets', []) if 'bin-win-cpu-x64.zip' in a.get('name', '')]; pathlib.Path(r'%LLAMA_DIR%\url.txt').write_text(urls[0], encoding='utf-8')"
    if errorlevel 1 exit /b 1

    set /p LLAMA_URL=<"%LLAMA_DIR%\url.txt"
    del /Q "%LLAMA_DIR%\url.txt" 2>nul

    echo [SG-MEDIATOR] Downloading llama.cpp CPU x64 release archive...
    curl.exe -fL -o "%ROOT_DIR%runtime\llama_cpu.zip" "%LLAMA_URL%"
    if errorlevel 1 exit /b 1

    echo [SG-MEDIATOR] Extracting llama-cli.exe and CPU runtime DLLs...
    tar.exe -xf "%ROOT_DIR%runtime\llama_cpu.zip" -C "%LLAMA_DIR%"
    if errorlevel 1 exit /b 1
    del /Q "%ROOT_DIR%runtime\llama_cpu.zip" 2>nul

    "%ROOT_DIR%runtime\python\python.exe" -c "import pathlib, shutil; root = pathlib.Path(r'%LLAMA_DIR%'); hits = list(root.rglob('llama-cli.exe')); (hits and hits[0].parent != root) and [shutil.move(str(p), str(root / p.name)) for p in hits[0].parent.iterdir() if not (root / p.name).exists()]"
)

REM Reuse local Ollama Qwen2.5-3B-Instruct-Q4_K_M GGUF blob if already cached on dev workstation; else download from HF
"%ROOT_DIR%runtime\python\python.exe" -c "exec('import os, pathlib, shutil\ndst = pathlib.Path(r\"%LLM_DIR%\\Qwen2.5-3B-Instruct-Q4_K_M.gguf\")\ntry:\n for d in [pathlib.Path(os.path.expanduser(\"~\"))/\".ollama\"/\"models\"/\"blobs\", pathlib.Path(r\"%LLM_DIR%\")/\"ollama\"/\"blobs\"]:\n  if not dst.exists() and d.exists():\n   for p in d.glob(\"sha256-*\"):\n    if 1800000000 < p.stat().st_size < 2100000000 and open(p, \"rb\").read(4) == b\"GGUF\": shutil.copy2(p, dst); break\nexcept Exception: pass')"

if not exist "%LLM_DIR%\Qwen2.5-3B-Instruct-Q4_K_M.gguf" (
    echo [SG-MEDIATOR] Downloading Qwen2.5-3B-Instruct-Q4_K_M.gguf ~1.93 GB...
    curl.exe -fL -o "%LLM_DIR%\Qwen2.5-3B-Instruct-Q4_K_M.gguf" https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
    if errorlevel 1 exit /b 1
)

echo.
echo [SG-MEDIATOR] Zero-socket Tier-5 llama-cli.exe + Qwen2.5-3B-Instruct-Q4_K_M.gguf ready!
echo Run run_sg_mediator.bat with --profile 16gb on the air-gapped PC.
endlocal
