@REM Set current working directory
cd /D "%~dp0"

@echo --------
@echo Create and activate venv
@echo --------
@echo
@python -m venv .\.venv
@call .\.venv\Scripts\activate.bat


@echo --------
@echo Install initial python libs and other software, takes a while...
@echo --------
@echo
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install einops fire diffusers accelerate peft huggingface-hub safetensors sentencepiece transformers tokenizers protobuf requests invisible-watermark spandrel

@echo --------
@echo Done!

pause