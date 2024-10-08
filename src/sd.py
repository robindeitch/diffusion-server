import os
from diffusers import FluxPipeline

from diffusers import DiffusionPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline, ControlNetModel, AutoencoderKL
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
def flatten(model: torch.nn.Module) -> list[torch.nn.Module]:
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


def seamless_tiling(pipeline:DiffusionPipeline) -> None:
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

def model_path(path_under_models_folder:str) -> str:
    models_base = "../../../models/"
    return os.path.join(models_base, path_under_models_folder)


def log_timing(prev_time:float, message:str) -> float:

    this_time = timer()
    print("********")
    print(f"{message} @ {this_time} (last step took {this_time - prev_time})")
    print("********")
    
    return this_time


def load_sdxl_with_controlnet(model_file:str) -> tuple[DiffusionPipeline, DiffusionPipeline]:

    vae_model_folder = model_path("hugging-face/madebyollin/sdxl-vae-fp16-fix")
    controlnet_model_folder = model_path("hugging-face/diffusers/controlnet-depth-sdxl-1.0")
    refiner_model_file = model_path("hugging-face/stabilityai/stable-diffusion-xl-refiner-1.0/sd_xl_refiner_1.0.safetensors")

    # initialize the models and pipeline
    prev_time = log_timing(0, "Loading ControlNetModel")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_folder,
        torch_dtype=dtype,
        local_files_only=True
    )

    prev_time = log_timing(prev_time, "Loading AutoencoderKL")
    vae = AutoencoderKL.from_pretrained(
        vae_model_folder,
        torch_dtype=dtype,
        local_files_only=True
    )

    # Create SDXL base pipeline
    prev_time = log_timing(prev_time, "Loading StableDiffusionXLControlNetPipeline")
    base_pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        model_file,
        config="../config/sdxl10",
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
        refiner_model_file,
        config="../config/sdxl10-refiner",
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

    prev_time = log_timing(prev_time, "Finished")

    return base_pipe, refiner_pipe

def generate_using_sdxl_with_controlnet(prompt:str, base_pipe:DiffusionPipeline, refiner_pipe:DiffusionPipeline, depth_image_file:str, output_image_file:str) -> None:

    negative_prompt = "low quality, bad quality, sketches"
    controlnet_conditioning_scale = 0.85

    prev_time = log_timing(0, "Loading image")
    image = load_image(depth_image_file)
    image = image.resize((1024, 512))

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
    image.save(output_image_file)
    prev_time = log_timing(prev_time, "Finished")


if __name__ == "__main__":
    prompt0 = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
    prompt1 = "Equirectangular projection. A photograph captures towering sci-fi buildings with cinematic grandeur. \
        The scene is bathed in black and white, with an orange accent color sparingly used to accentuate the architectural details. \
        Meticulously rendered, showcasing intricate textures and breathtaking scale. Cinematic masterpiece on ArtStation evokes a sense of awe and grandeur"
    prompt2 = "Equirectangular projection. A photograph captures towering sci-fi buildings with cinematic grandeur. \
        The scene is moody and threatening with an orange accent color sparingly used to accentuate the architectural details. \
        Meticulously rendered, showcasing intricate textures and breathtaking scale. Cinematic masterpiece on ArtStation evokes a sense of awe and grandeur"
    prompt3 = "A photograph captures towering sci-fi buildings with cinematic grandeur. \
        The scene is moody and threatening with an orange accent color sparingly used to accentuate the architectural details. \
        Meticulously rendered, showcasing intricate textures and breathtaking scale. Cinematic masterpiece on ArtStation evokes a sense of awe and grandeur"

    depth_image = "../../diffusion-server-files/input-depth.png"
    
    sdxl_model_mohawk = model_path("civitai/sdxl_10/MOHAWK_v20.safetensors")
    sdxl_model_cardos = model_path("civitai/sdxl_10/cardosXL_v10.safetensors")

    output_image1 = "../../diffusion-server-files/sdxl_controlnet_example1.png"
    output_image2 = "../../diffusion-server-files/sdxl_controlnet_example2.png"
    output_image3 = "../../diffusion-server-files/sdxl_controlnet_example3.png"

    base_pipe, refiner_pipe = load_sdxl_with_controlnet(model_file=sdxl_model_mohawk)
    generate_using_sdxl_with_controlnet(prompt=prompt0, base_pipe=base_pipe, refiner_pipe=refiner_pipe, depth_image_file=depth_image, output_image_file=output_image1)
    generate_using_sdxl_with_controlnet(prompt=prompt2, base_pipe=base_pipe, refiner_pipe=refiner_pipe, depth_image_file=depth_image, output_image_file=output_image2)
    generate_using_sdxl_with_controlnet(prompt=prompt3, base_pipe=base_pipe, refiner_pipe=refiner_pipe, depth_image_file=depth_image, output_image_file=output_image3)
