@echo off
setlocal
REM OPTIONAL Tier-2.5B GLiNER-BioMed INT8 ONNX Span Extractor Setup (< 380 MB RAM).
REM Run on a workstation WITH internet; copy models_onnx_int8\gliner_biomed_int8\ to the offline PC.
REM When absent, SG-MEDIATOR automatically uses its built-in clinical span chunker + INT8 SapBERT VQ linker.

set "ROOT_DIR=%~dp0"
set "OUT_DIR=%ROOT_DIR%models_onnx_int8\gliner_biomed_int8"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo [SG-MEDIATOR] Downloading pre-quantized GLiNER-BioMed INT8 ONNX artifacts (onnx-community/gliner-biomed-bi-small-v1.0)...
curl.exe -fL -o "%OUT_DIR%\model_quantized.onnx" https://huggingface.co/onnx-community/gliner-biomed-bi-small-v1.0/resolve/main/onnx/model_quantized.onnx
if errorlevel 1 (
    echo [SG-MEDIATOR] Note: Direct pre-quantized ONNX mirror unavailable; see antigravity_kit or export via optimum-cli on a PyTorch dev machine.
    exit /b 1
)
curl.exe -fL -o "%OUT_DIR%\tokenizer.json" https://huggingface.co/Ihor/gliner-biomed-bi-small-v1.0/resolve/main/tokenizer.json
if errorlevel 1 exit /b 1
curl.exe -fL -o "%OUT_DIR%\gliner_config.json" https://huggingface.co/Ihor/gliner-biomed-bi-small-v1.0/resolve/main/gliner_config.json
if errorlevel 1 exit /b 1

echo [SG-MEDIATOR] Optional GLiNER-BioMed INT8 ONNX slot ready at %OUT_DIR%!
endlocal
