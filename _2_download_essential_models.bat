@REM Set current working directory
cd /D "%~dp0\.models"

curl -L -O https://huggingface.co/stabilityai/sdxl-vae/resolve/main/diffusion_pytorch_model.safetensors --output-dir .\stabilityai\sdxl-vae
curl -L -O https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors --output-dir .\madebyollin\sdxl-vae-fp16-fix
curl -L -O https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors --output-dir .\stabilityai\stable-diffusion-xl-refiner-1.0
curl -L https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors --output .\xinsir\controlnet-union-sdxl-1.0\diffusion_pytorch_model.safetensors

@echo --------
@echo Done!

pause