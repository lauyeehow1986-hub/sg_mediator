@echo off
setlocal
REM Downloads the INT8 ONNX biomedical encoder (SapBERT-from-PubMedBERT-fulltext) used by Head 1 (VQ)
REM into models_onnx_int8\sapbert_pubmedbert_int8 and builds the K=16,384 spherical VQ codebook.
REM Run on a workstation WITH internet, then copy the whole folder to the offline target.
set "ROOT_DIR=%~dp0"
set "MDIR=%ROOT_DIR%models_onnx_int8\sapbert_pubmedbert_int8"
if not exist "%MDIR%" mkdir "%MDIR%"

echo [SG-MEDIATOR] Downloading INT8 ONNX SapBERT biomedical weights...
curl.exe -fL -o "%MDIR%\model_quantized.onnx" https://huggingface.co/Xenova/SapBERT-from-PubMedBERT-fulltext/resolve/main/onnx/model_quantized.onnx
if errorlevel 1 exit /b 1
curl.exe -fL -o "%MDIR%\tokenizer.json" https://huggingface.co/Xenova/SapBERT-from-PubMedBERT-fulltext/resolve/main/tokenizer.json
if errorlevel 1 exit /b 1

if exist "%ROOT_DIR%runtime\python\python.exe" (
    set "PY_EXE=%ROOT_DIR%runtime\python\python.exe"
) else (
    set "PY_EXE=python"
)

echo [SG-MEDIATOR] Training K=16384 spherical biomedical VQ codebook...
"%PY_EXE%" "%ROOT_DIR%scripts\train_vq_codebook.py"
if errorlevel 1 exit /b 1

echo [SG-MEDIATOR] Done. Biomedical encoder + K=16384 VQ codebook ready in %MDIR%.
endlocal
