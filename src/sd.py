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

class LoraInfo:
    def __init__(self, model:str, key:str) -> None:
        self.model = model
        self.key = key

def lorafy(prompt:str, loras:list[LoraInfo]) -> str:
    lora_str = ", ".join([l.key for l in loras])
    return f"{lora_str}, {prompt}"

# --------------------------

def load_sdxl_with_controlnet(model_file:str, loras:list[LoraInfo], lora_weights:list[float]) -> tuple[DiffusionPipeline, DiffusionPipeline]:

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

    prev_time = log_timing(prev_time, f"Loading AutoencoderKL from {vae_model_folder}")
    vae = AutoencoderKL.from_pretrained(
        vae_model_folder,
        torch_dtype=dtype,
        local_files_only=True
    )

    # Create SDXL base pipeline
    prev_time = log_timing(prev_time, f"Loading StableDiffusionXLControlNetPipeline from {model_file}")
    base_pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        model_file,
        config="../config/sdxl10",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        local_dir_use_symlinks=False,
        local_files_only=True
    )
    if len(loras) > 0:
        names = [f'lora{index}' for index, _ in enumerate(loras)]

        for index, lora in enumerate(loras):
            prev_time = log_timing(prev_time, f"Loading LoRa from {lora.model} ({names[index]})")
            base_pipe.load_lora_weights(lora.model, adapter_name=names[index])
        
        prev_time = log_timing(prev_time, f"Setting LoRa weights : {', '.join(names)} -> {', '.join([str(w) for w in lora_weights])}")
        base_pipe.set_adapters(names, adapter_weights=lora_weights)

    base_pipe.enable_model_cpu_offload()

    # Create SDXL refiner pipeline
    if len(loras) > 0:
        refiner_pipe = None
    else:
        prev_time = log_timing(prev_time, f"Loading StableDiffusionXLImg2ImgPipeline from {refiner_model_file}")
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

def generate_using_sdxl_with_controlnet(prompt:str, negative_prompt:str, base_pipe:DiffusionPipeline, refiner_pipe:DiffusionPipeline, depth_image_file:str, output_image_file:str, lora_scale:float = 0) -> None:

    controlnet_conditioning_scale = 0.85

    prev_time = log_timing(0, f"Loading depth image from {depth_image_file}")
    image = load_image(depth_image_file)
    image = image.resize((1024, 512))

    # generate image
    inference_steps = 50

    if refiner_pipe is not None:
        prev_time = log_timing(prev_time, "Generating base image")
        refiner_start_percentage = 0.75
        image = base_pipe(
            prompt,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            image=[image],
            num_inference_steps=inference_steps,
            denoising_end=refiner_start_percentage,
            generator=torch.manual_seed(0),
            output_type="latent"
        ).images

        prev_time = log_timing(prev_time, "Refining image")
        image = refiner_pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=inference_steps,
            denoising_start=refiner_start_percentage,
            generator=torch.manual_seed(0)
        ).images[0]
    else:
        prev_time = log_timing(prev_time, "Generating image with LoRa")
        image = base_pipe(
            prompt,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            image=[image],
            num_inference_steps=inference_steps,
            cross_attention_kwargs={"scale": lora_scale},
            generator=torch.manual_seed(0)
        ).images[0]

    prev_time = log_timing(prev_time, f"Saving image to {output_image_file}")
    image.save(output_image_file)
    prev_time = log_timing(prev_time, "Finished")


if __name__ == "__main__":

    negative_prompt = "low quality, bad quality, sketches"

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

    sdxl_lora_scifistyle = LoraInfo( model_path("civitai/sdxl_10_lora/scifi_buildings_sdxl_lora-scifistyle-cinematic scifi.safetensors"), '#scifistyle, cinematic scifi' )
    sdxl_lora_moebius = LoraInfo( model_path("civitai/sdxl_10_lora/Moebius comic book_SDXL.safetensors"), 'FRESHIDESH Moebius comic book' )
    sdxl_lora_eboy = LoraInfo( model_path("civitai/sdxl_10_lora/Eboy_Pixelart_Style_XL.safetensors"), 'eboy style' )
    sdxl_lora_anime = LoraInfo( model_path("civitai/sdxl_10_lora/Anime_Sketch_SDXL.safetensors"), '(Pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch), unfinished, hatching (texture)' )
    sdxl_lora_woodcut = LoraInfo( model_path("civitai/sdxl_10_lora/Landscape printsSDXL.safetensors"), 'Landscape woodcut prints' )
    sdxl_lora_ghibli = LoraInfo( model_path("civitai/sdxl_10_lora/Studio Ghibli Style.safetensors"), 'Studio Ghibli Style' )
    sdxl_lora_chalkboard = LoraInfo( model_path("civitai/sdxl_10_lora/SDXL_ChalkBoardDrawing_LoRA_r8.safetensors"), 'ChalkBoardDrawing' )

    def img(version:str) -> str:
        return f"../../diffusion-server-files/sdxl_controlnet_{version}.png"

    base_pipe, refiner_pipe = load_sdxl_with_controlnet(model_file=sdxl_model_mohawk, loras=[sdxl_lora_ghibli, sdxl_lora_woodcut], lora_weights=[0.8, 0.1])
    #generate_using_sdxl_with_controlnet(prompt=prompt3, negative_prompt=negative_prompt, base_pipe=base_pipe, refiner_pipe=refiner_pipe, depth_image_file=depth_image, output_image_file=img('ghibli2-control'))
    #generate_using_sdxl_with_controlnet(prompt=lorafy(prompt3, [sdxl_lora_ghibli, sdxl_lora_woodcut]), negative_prompt=negative_prompt, base_pipe=base_pipe, refiner_pipe=refiner_pipe, lora_scale=0.0, depth_image_file=depth_image, output_image_file=img('ghibli2-0'))
    generate_using_sdxl_with_controlnet(prompt=lorafy(prompt3, [sdxl_lora_ghibli, sdxl_lora_woodcut]), negative_prompt=negative_prompt, base_pipe=base_pipe, refiner_pipe=refiner_pipe, lora_scale=1.0, depth_image_file=depth_image, output_image_file=img('ghibli2-1x'))
