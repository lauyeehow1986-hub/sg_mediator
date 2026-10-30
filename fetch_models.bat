@echo off
REM Downloads the INT8 ONNX encoder used by Head 1 (VQ) into models_onnx_int8\minilm_l6_int8.
REM Run on a workstation WITH internet, then copy the whole folder to the offline target.
set "ROOT_DIR=%~dp0"
set "MDIR=%ROOT_DIR%models_onnx_int8\minilm_l6_int8"
if not exist "%MDIR%" mkdir "%MDIR%"
curl.exe -fL -o "%MDIR%\model_quantized.onnx" https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx
curl.exe -fL -o "%MDIR%\tokenizer.json" https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json
echo Done.
