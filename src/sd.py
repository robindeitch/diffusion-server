import os
from diffusers import FluxPipeline

from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline, ControlNetModel, AutoencoderKL
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.utils import load_image
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
import numpy as np
import torch
import cv2
from PIL import Image

from timeit import default_timer as timer

dtype = torch.bfloat16

# ---------------------
# Seamless tiling from here : https://github.com/huggingface/diffusers/issues/556
def flatten(model: torch.nn.Module):
    """
    Recursively flattens the model to retrieve all layers.
    """
    children = list(model.children())
    flattened = []

    if children == []:
        return model

    for child in children:
        try:
            flattened.extend(flatten(child))
        except TypeError:
            flattened.append(flatten(child))
    return flattened


def seamless_tiling(pipeline):
    """
    Enables seamless tiling for specific layers in the pipeline.
    """
    targets = [pipeline.vae, pipeline.text_encoder, pipeline.unet]

    if hasattr(pipeline, "text_encoder_2"):
        targets.append(pipeline.text_encoder_2)
    if pipeline.image_encoder is not None:
        targets.append(pipeline.image_encoder)

    layers = [
        layer
        for target in targets if target is not None
        for layer in flatten(target)
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
    ]

    for layer in layers:
        layer.padding_mode = "circular"
        layer.padding = (1,0)

# ---------------------

#model = "../../models/black-forest-labs/Flux.1-schnell/"
def model_path(path_under_models_folder):
    models_base = "../../../models/"
    return os.path.join(models_base, path_under_models_folder)

def log_timing(prev_time, message):

    this_time = timer()
    print("********")
    print(f"{message} @ {this_time} (last step took {this_time - prev_time})")
    print("********")
    
    return this_time

def sdxl_controlnet_example():

    #prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
    prompt = "Equirectangular projection. A photograph captures towering sci-fi buildings with cinematic grandeur. \
        The scene is bathed in black and white, with an orange accent color sparingly used to accentuate the architectural details. \
        Meticulously rendered, showcasing intricate textures and breathtaking scale. Cinematic masterpiece on ArtStation evokes a sense of awe and grandeur"
    prompt = "Equirectangular projection. A photograph captures towering sci-fi buildings with cinematic grandeur. \
        The scene is moody and threatening with an orange accent color sparingly used to accentuate the architectural details. \
        Meticulously rendered, showcasing intricate textures and breathtaking scale. Cinematic masterpiece on ArtStation evokes a sense of awe and grandeur"
    prompt = "A photograph captures towering sci-fi buildings with cinematic grandeur. \
        The scene is moody and threatening with an orange accent color sparingly used to accentuate the architectural details. \
        Meticulously rendered, showcasing intricate textures and breathtaking scale. Cinematic masterpiece on ArtStation evokes a sense of awe and grandeur"
    negative_prompt = "low quality, bad quality, sketches"

    prev_time = log_timing(0, "Loading image")
    image = load_image("../../diffusion-server-files/input-depth.png")
    image = image.resize((1024, 512))

    # initialize the models and pipeline
    prev_time = log_timing(prev_time, "Loading ControlNetModel")
    controlnet_conditioning_scale = 0.85
    controlnet = ControlNetModel.from_pretrained(model_path("hugging-face/diffusers/controlnet-depth-sdxl-1.0"), torch_dtype=dtype)

    prev_time = log_timing(prev_time, "Loading AutoencoderKL")
    vae = AutoencoderKL.from_pretrained(model_path("hugging-face/madebyollin/sdxl-vae-fp16-fix"), torch_dtype=dtype)

    # Create SDXL base pipeline
    prev_time = log_timing(prev_time, "Loading StableDiffusionXLControlNetPipeline")
    base_pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        model_path("civitai/sdxl_10/MOHAWK_v20.safetensors"), 
        config=model_path("civitai/sdxl_10"),
        controlnet=controlnet, 
        vae=vae, 
        torch_dtype=dtype,
        local_dir_use_symlinks=False,
        local_files_only=True
    )
    base_pipe.enable_model_cpu_offload()

    # Create SDXL refiner pipeline
    prev_time = log_timing(prev_time, "Loading StableDiffusionXLImg2ImgPipeline")
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path("hugging-face/stabilityai/stable-diffusion-xl-refiner-1.0/sd_xl_refiner_1.0.safetensors"),
        config=model_path("hugging-face/stabilityai/stable-diffusion-xl-refiner-1.0"),
        vae=base_pipe.vae,
        text_encoder_2=base_pipe.text_encoder_2,
        torch_dtype=dtype
    )
    refiner_pipe.enable_model_cpu_offload()

    # Compiler models - doesn't seem to work, might need to pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu121"
    #prev_time = log_timing(prev_time, "Compiling StableDiffusionXLControlNetPipeline")
    #base_pipe.unet = torch.compile(base_pipe.unet)
    #prev_time = log_timing(prev_time, "Compiling StableDiffusionXLImg2ImgPipeline")
    #refiner_pipe.unet = torch.compile(refiner_pipe.unet)

    # Make both pipelins tile seamlessly - does both directions, need to look at eg https://github.com/jn-jairo/jn_node_suite_comfyui/blob/master/nodes/patch.py
    #seamless_tiling(pipeline=base_pipe)
    #seamless_tiling(pipeline=refiner_pipe)

    # generate image
    prev_time = log_timing(prev_time, "Generating base image")
    inference_steps = 50
    refiner_start_percentage = 0.75
    image = base_pipe(
        prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        image=[image],
        num_inference_steps=inference_steps,
        denoising_end=refiner_start_percentage,
        output_type="latent"
    ).images

    prev_time = log_timing(prev_time, "Refining image")
    image = refiner_pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=inference_steps,
        denoising_start=refiner_start_percentage,
    ).images[0]

    prev_time = log_timing(prev_time, "Saving image")
    image.save("../../diffusion-server-files/sdxl_controlnet_example.png")
    prev_time = log_timing(prev_time, "Finished")

def sd_controlnet():

    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel

    prev_time = log_timing(0, "Creating ControlNetModel")
    controlnet = ControlNetModel.from_single_file(model_path("hugging-face/Illyasviel/ControlNet-v1-1/control_v11f1p_sd15_depth.pth"), local_files_only=True)

    prev_time = log_timing(prev_time, "Creating StableDiffusionControlNetPipeline")
    ckpt_path = model_path("hugging-face/stable-diffusion-v1.5/stable-diffusion-v1.5/v1-5-pruned-emaonly.safetensors")
    config_path = model_path("")
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(ckpt_path, controlnet=controlnet, local_files_only=True)

    prev_time = log_timing(prev_time, "Creating image")
    prompt = "A cat took a fish and running in a market"
    image = pipe(
        prompt,
        guidance_scale=3.5,
        width=512,
        height=512,
        num_inference_steps=4,
    ).images[0]

    prev_time = log_timing(prev_time, "Saving image")
    image.save("cat.png")

    prev_time = log_timing(prev_time, "Done")

def flux_schnell():

    prev_time = log_timing(0, "Creating FlowMatchEulerDiscreteScheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(models_base, "hugging-face/black-forest-labs/Flux.1-schnell/scheduler"))

    prev_time = log_timing(prev_time, "Creating CLIPTextModel")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)

    prev_time = log_timing(prev_time, "Creating CLIPTokenizer")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)

    prev_time = log_timing(prev_time, "Creating T5EncoderModel")
    text_encoder_2 = T5EncoderModel.from_pretrained(os.path.join(models_base, "hugging-face/black-forest-labs/Flux.1-schnell/text_encoder_2"), torch_dtype=dtype)

    prev_time = log_timing(prev_time, "Creating T5TokenizerFast")
    tokenizer_2 = T5TokenizerFast.from_pretrained(os.path.join(models_base, "hugging-face/black-forest-labs/Flux.1-schnell/tokenizer_2"), torch_dtype=dtype)

    prev_time = log_timing(prev_time, "Creating AutoencoderKL")
    vae = AutoencoderKL.from_pretrained(os.path.join(models_base, "hugging-face/black-forest-labs/Flux.1-schnell/vae"), torch_dtype=dtype)

    prev_time = log_timing(prev_time, "Creating FluxTransformer2DModel")
    transformer = FluxTransformer2DModel.from_single_file(os.path.join(models_base, "hugging-face/kijai/flux1-schnell-fp8-e4m3fn.safetensors"), torch_dtype=dtype)

    prev_time = log_timing(prev_time, "Creating FluxPipeline")
    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None,
    )
    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer = transformer
    #pipe.enable_model_cpu_offload()
    pipe.to("cuda")

    prev_time = log_timing(prev_time, "Creating image")
    prompt = "A cat took a fish and running in a market"
    image = pipe(
        prompt,
        guidance_scale=3.5,
        width=512,
        height=512,
        num_inference_steps=4,
    ).images[0]

    prev_time = log_timing(prev_time, "Saving image")
    image.save("cat.png")

    prev_time = log_timing(prev_time, "Done")

if __name__ == "__main__":
    sdxl_controlnet_example()