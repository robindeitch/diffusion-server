@REM Set current working directory
cd /D "%~dp0\.models"

curl -L -O https://huggingface.co/stabilityai/sdxl-vae/resolve/main/diffusion_pytorch_model.safetensors --output-dir .\stabilityai\sdxl-vae
curl -L -O https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors --output-dir .\stabilityai\stable-diffusion-xl-refiner-1.0
curl -L -O https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors --output-dir .\diffusers\controlnet-depth-sdxl-1.0

@echo --------
@echo Done!

pause