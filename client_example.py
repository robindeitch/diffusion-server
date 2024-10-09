import os
from random import randint
from PIL import Image
import time
from sdxl_client import SDXLClient

cwd = os.getcwd()
models_base = os.path.join(cwd, "../../models/")

def model_path(path_under_models_folder:str) -> str:
    return os.path.join(models_base, path_under_models_folder)

sdxl_model_mohawk = model_path("civitai/sdxl_10/MOHAWK_v20.safetensors")
#sdxl_model_cardos = model_path("civitai/sdxl_10/cardosXL_v10.safetensors")
#sdxl_lightning_model_wildcard = model_path("civitai/sdxl_lightning/wildcardxXLLIGHTNING_wildcardxXL.safetensors")
#sdxl_lightning_model_base = model_path("hugging-face/ByteDance/SDXL-Lightning/sdxl_lightning_4step.safetensors")

sdxl_lora_offset_noise = (model_path("hugging-face/stabilityai/stable-diffusion-xl-base-1.0/sd_xl_offset_example-lora_1.0.safetensors"), "")
sdxl_lora_scifistyle = (model_path("civitai/sdxl_10_lora/scifi_buildings_sdxl_lora-scifistyle-cinematic scifi.safetensors"), "#scifistyle, cinematic scifi")
#sdxl_lora_moebius = ( model_path("civitai/sdxl_10_lora/Moebius comic book_SDXL.safetensors"), "FRESHIDESH Moebius comic book" )
#sdxl_lora_eboy = ( model_path("civitai/sdxl_10_lora/Eboy_Pixelart_Style_XL.safetensors"), "eboy style" )
#sdxl_lora_anime = ( model_path("civitai/sdxl_10_lora/Anime_Sketch_SDXL.safetensors"), "(Pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch), unfinished, hatching (texture)" )
#sdxl_lora_woodcut = ( model_path("civitai/sdxl_10_lora/Landscape printsSDXL.safetensors"), "Landscape woodcut prints" )
#sdxl_lora_ghibli = ( model_path("civitai/sdxl_10_lora/Studio Ghibli Style.safetensors"), "Studio Ghibli Style" )
#sdxl_lora_chalkboard = ( model_path("civitai/sdxl_10_lora/SDXL_ChalkBoardDrawing_LoRA_r8.safetensors"), "ChalkBoardDrawing" )


if __name__ == "__main__":

    # Connect to the server and load a model + LoRAs
    client = SDXLClient()
    client.init(sdxl_model_mohawk, [sdxl_lora_offset_noise, sdxl_lora_scifistyle], [0.2, 0.7])

    # Start generating
    seed = randint(1, 2147483647)
    steps = 15
    prompt_guidance=7.5
    depth_image_influence = 0.85
    lora_overall_influence = 1.0
    depth_image = Image.open(os.path.join(cwd, "../diffusion-server-files/input-depth.png"))

    prompt = "Equirectangular projection. A photograph captures towering sci-fi buildings with cinematic grandeur. \
        The scene is bathed in black and white, with an orange accent color sparingly used to accentuate the architectural details. \
        Meticulously rendered, showcasing intricate textures and breathtaking scale. Cinematic masterpiece on ArtStation evokes a sense of awe and grandeur"

    negative_prompt = "low quality, bad quality, sketches, blurry, jpeg artifacts"

    # Syncronous example
    image = client.generate_panorama(prompt, negative_prompt, seed, steps, prompt_guidance, depth_image, depth_image_influence, lora_overall_influence)
    image.show()

    # Async example
    def callback(id:int, image:Image):
        print(f"Id {id} has finished")
        image.show()

    id1 = client.queue_panorama(callback, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image, depth_image_influence, 0.0 * lora_overall_influence)
    id2 = client.queue_panorama(callback, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image, depth_image_influence, 0.5 * lora_overall_influence)
    id3 = client.queue_panorama(callback, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image, depth_image_influence, 1.0 * lora_overall_influence)
    
    while True:
        time.sleep(5)
        print("...")
