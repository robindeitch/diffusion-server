from utils import LoraInfo, log_timing
from diffusers import DiffusionPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline, ControlNetModel, AutoencoderKL, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from PIL import Image
import torch

dtype = torch.float16

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

        controlnet_model_folder = "../.models/diffusers/controlnet-depth-sdxl-1.0"
        #vae_model_folder = "../.models/stabilityai/sdxl-vae"
        vae_model_folder = "../.models/madebyollin/sdxl-vae-fp16-fix"
        refiner_model_file = "../.models/stabilityai/stable-diffusion-xl-refiner-1.0/sd_xl_refiner_1.0.safetensors"

        # initialize the models and pipeline
        prev_time = log_timing(0, "Loading ControlNetModel")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_folder,
            variant="fp16",
            torch_dtype=dtype,
            local_files_only=True
        )
        controlnet.to("cuda")

        prev_time = log_timing(prev_time, f"Loading AutoencoderKL from {vae_model_folder}")
        vae = AutoencoderKL.from_pretrained(
            vae_model_folder,
            torch_dtype=dtype,
            local_files_only=True
        )
        vae.to("cuda")

        # Create SDXL pipeline
        prev_time = log_timing(prev_time, f"Loading StableDiffusionXLControlNetPipeline from {model_file}")
        base_pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            model_file,
            variant="fp16",
            config="../config/sdxl10",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
            local_dir_use_symlinks=False,
            local_files_only=True,
            add_watermarker=False
        )
        # https://huggingface.co/docs/diffusers/v0.26.2/en/api/schedulers/overview#schedulers
        base_pipe.scheduler = EulerDiscreteScheduler.from_config(base_pipe.scheduler.config, timestep_spacing="trailing")
        #base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        if len(loras) > 0:
            names = [f'lora{index}' for index, _ in enumerate(loras)]

            for index, lora in enumerate(loras):
                prev_time = log_timing(prev_time, f"Loading LoRa from {lora.model} ({names[index]})")
                base_pipe.load_lora_weights(lora.model, adapter_name=names[index])
            
            prev_time = log_timing(prev_time, f"Setting LoRa weights : {', '.join(names)} -> {', '.join([str(w) for w in lora_weights])}")
            base_pipe.set_adapters(names, adapter_weights=lora_weights)

            self.lora_prompt = ", ".join([lora.key for lora in loras])

        base_pipe.to("cuda")

        # Create SDXL refiner pipeline
        refiner_pipe = None
        # if len(loras) == 0:
        #     prev_time = log_timing(prev_time, f"Loading StableDiffusionXLImg2ImgPipeline from {refiner_model_file}")
        #     refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        #         refiner_model_file,
        #         config="../config/sdxl10-refiner",
        #         vae=base_pipe.vae,
        #         text_encoder_2=base_pipe.text_encoder_2,
        #         torch_dtype=dtype,
        #         add_watermarker=False
        #     )
        #     refiner_pipe.enable_model_cpu_offload()

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


    def generate_panorama(self, prompt:str, negative_prompt:str, seed:int, steps:int, prompt_guidance:float, depth_image:Image, depth_image_influence:float, lora_overall_influence:float = 0) -> Image:

        prompt = ", ".join([self.lora_prompt, prompt])

        prev_time = log_timing(0, f"Resizing depth image from ({depth_image.width}, {depth_image.height}) to (1024, 512)")
        depth_image = depth_image.resize((1024, 512))

        # generate image
        generator = torch.manual_seed(seed)
        image:Image = None
        if self.refiner_pipe is not None:
            prev_time = log_timing(prev_time, f"Generating base image with prompt : {prompt}")
            refiner_start_percentage = 0.75
            image = self.base_pipe(
                prompt,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=depth_image_influence,
                image=[depth_image],
                num_inference_steps=steps,
                denoising_end=refiner_start_percentage,
                guidance_scale=prompt_guidance,
                generator=generator,
                output_type="latent"
            ).images

            prev_time = log_timing(prev_time, "Refining image")
            image = self.refiner_pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=steps,
                denoising_start=refiner_start_percentage,
                generator=generator
            ).images[0]
        else:
            prev_time = log_timing(prev_time, f"Generating image with LoRA and prompt {prompt}")
            image = self.base_pipe(
                prompt,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=depth_image_influence,
                image=[depth_image],
                num_inference_steps=steps,
                cross_attention_kwargs={"scale": lora_overall_influence},
                guidance_scale=prompt_guidance,
                original_size=(1024, 512),
                target_size=(1024, 512),
                generator=generator
            ).images[0]

        prev_time = log_timing(prev_time, "Finished")
        return image
