@REM Set current working directory
cd /D "%~dp0\.models"

curl -L -O https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors --output-dir .\madebyollin\sdxl-vae-fp16-fix
curl -L https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors --output .\xinsir\controlnet-union-sdxl-1.0\diffusion_pytorch_model.safetensors
curl -L -O https://huggingface.co/skbhadra/ClearRealityV1/resolve/main/4x-ClearRealityV1.pth --output-dir .\skbhadra\ClearRealityV1

@echo --------
@echo Done!

pause