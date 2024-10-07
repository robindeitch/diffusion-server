@REM Prerequisites :
@REM   Miniconda @ 24.7.1 - https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Windows-x86_64.exe
@REM   MSVC++Redist @ 14.36.32530.0 - https://download.visualstudio.microsoft.com/download/pr/eaab1f82-787d-4fd7-8c73-f782341a0c63/917C37D816488545B70AFFD77D6E486E4DD27E2ECE63F6BBAAF486B178B2B888/VC_redist.x64.exe
@REM     (versions here - https://github.com/abbodi1406/vcredist/blob/master/source_links/README.md)
@REM   NVIDIA CUDA Toolkit 12.1.0 - https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe
@REM                    or 12.1.1 - https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_531.14_windows.exe

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
python -m pip install einops fire diffusers accelerate huggingface-hub safetensors sentencepiece transformers tokenizers protobuf requests invisible-watermark

@echo --------
@echo Done!

pause