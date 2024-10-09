from utils import model_path, LoraInfo, log_timing
from diffusers import DiffusionPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline, ControlNetModel, AutoencoderKL
from diffusers import AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch

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

# --------------------------

class SDXL:

    lora_prompt:str = ''
    base_pipe: DiffusionPipeline = None
    refiner_pipe: DiffusionPipeline = None

    def __init__(self, model_file:str, loras:list[LoraInfo], lora_weights:list[float]) -> tuple[DiffusionPipeline, DiffusionPipeline]:

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

            self.lora_prompt = ", ".join([lora.key for lora in loras])

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

        self.base_pipe = base_pipe
        self.refiner_pipe = refiner_pipe


    def generate_using_depth(self, prompt:str, negative_prompt:str, depth_image_file:str, lora_scale:float = 0) -> Image:

        controlnet_conditioning_scale = 0.85

        prompt = ", ".join([self.lora_prompt, prompt])

        prev_time = log_timing(0, f"Loading depth image from {depth_image_file}")
        image = load_image(depth_image_file)
        image = image.resize((1024, 512))

        # generate image
        inference_steps = 50

        if self.refiner_pipe is not None:
            prev_time = log_timing(prev_time, f"Generating base image with prompt : {prompt}")
            refiner_start_percentage = 0.75
            image = self.base_pipe(
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
            image = self.refiner_pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=inference_steps,
                denoising_start=refiner_start_percentage,
                generator=torch.manual_seed(0)
            ).images[0]
        else:
            prev_time = log_timing(prev_time, f"Generating image with LoRA and prompt {prompt}")
            image = self.base_pipe(
                prompt,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                image=[image],
                num_inference_steps=inference_steps,
                cross_attention_kwargs={"scale": lora_scale},
                generator=torch.manual_seed(0)
            ).images[0]

        prev_time = log_timing(prev_time, "Finished")
        return image
