import os, sys
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
sdxl_lightning_model_wildcard = model_path("civitai/sdxl_lightning/wildcardxXLLIGHTNING_wildcardxXL.safetensors")
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

    # Connect to the server and start the worker thread
    client = SDXLClient()
    client.start()

    # Load a model + LoRAs
    client.init(sdxl_lightning_model_wildcard)

    # Start generating
    seed = randint(1, 2147483647)
    steps = 8
    prompt_guidance=3.5
    depth_image_influence = 0.65
    lora_overall_influence = 1.0
    depth_image_file = os.path.join(cwd, "./client-example-depth.png")

    prompt = "((masterpiece)), (cinematic), Equirectangular projection, 360 degree image, photography, A bleached white photograph captures \
        towering modernist skyscrapers with cinematic grandeur. The scene has an orange accent color sparingly used to accentuate the architectural details. \
        Meticulously rendered, showcasing intricate textures and breathtaking scale, it evokes a sense of awe and grandeur. "

    negative_prompt = "text, watermark, uniform,  ugly, high contrast, jpeg, (worst quality, low quality, lowres, low details, overexposed, underexposed, \
        grayscale, bw,  bad art:1.4), (font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), \
            poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (amateur:1.3), cave"

    # Syncronous example
    output_file = os.path.join(cwd, "test_output.jpg")
    image_file = client.generate_panorama(output_file, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image_file, depth_image_influence, lora_overall_influence)
    image = Image.open(image_file)
    image.show()

    sys.exit(0)

    # Async example

    task_ids = []

    def callback(id:int, image_file:str):
        print(f"Id {id} has finished")
        img = Image.open(image_file)
        img.show()
        task_ids.remove(id)

    output_file_1 = os.path.join(cwd, "test_output_1.png")
    output_file_2 = os.path.join(cwd, "test_output_2.png")
    output_file_3 = os.path.join(cwd, "test_output_3.png")
    task_ids.append( client.queue_panorama(output_file_1, callback, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image_file, depth_image_influence, 0.0 * lora_overall_influence) )
    task_ids.append( client.queue_panorama(output_file_2, callback, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image_file, depth_image_influence, 0.5 * lora_overall_influence) )
    task_ids.append( client.queue_panorama(output_file_3, callback, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image_file, depth_image_influence, 1.0 * lora_overall_influence) )
    
    while True and len(task_ids) > 0:
        time.sleep(5)
        print("...")

    # Stop the client's worker thread
    client.stop()
