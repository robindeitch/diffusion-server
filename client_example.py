import os, sys, io
import xmlrpc.client
from PIL import Image

cwd = os.getcwd()
models_base = os.path.join(cwd, "../../models/")

def image_from_png_bytes(data:bytes) -> Image:
    b =  io.BytesIO(data)
    return Image.open(b)

def image_to_png_bytes(img:Image) -> bytes:
    b = io.BytesIO()
    img.save(b, 'PNG')
    return b.getvalue()


if __name__ == "__main__":

    negative_prompt = "low quality, bad quality, sketches, blurry, jpeg artifacts"

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

    def model_path(path_under_models_folder:str) -> str:
        return os.path.join(models_base, path_under_models_folder)

    sdxl_model_mohawk = model_path("civitai/sdxl_10/MOHAWK_v20.safetensors")
    sdxl_model_cardos = model_path("civitai/sdxl_10/cardosXL_v10.safetensors")
    sdxl_lightning_model_wildcard = model_path("civitai/sdxl_lightning/wildcardxXLLIGHTNING_wildcardxXL.safetensors")
    sdxl_lightning_model_base = model_path("hugging-face/ByteDance/SDXL-Lightning/sdxl_lightning_4step.safetensors")

    sdxl_lora_offset_noise = (model_path("hugging-face/stabilityai/stable-diffusion-xl-base-1.0/sd_xl_offset_example-lora_1.0.safetensors"), "")
    sdxl_lora_scifistyle = (model_path("civitai/sdxl_10_lora/scifi_buildings_sdxl_lora-scifistyle-cinematic scifi.safetensors"), "#scifistyle, cinematic scifi")
    sdxl_lora_moebius = ( model_path("civitai/sdxl_10_lora/Moebius comic book_SDXL.safetensors"), "FRESHIDESH Moebius comic book" )
    sdxl_lora_eboy = ( model_path("civitai/sdxl_10_lora/Eboy_Pixelart_Style_XL.safetensors"), "eboy style" )
    sdxl_lora_anime = ( model_path("civitai/sdxl_10_lora/Anime_Sketch_SDXL.safetensors"), "(Pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch), unfinished, hatching (texture)" )
    sdxl_lora_woodcut = ( model_path("civitai/sdxl_10_lora/Landscape printsSDXL.safetensors"), "Landscape woodcut prints" )
    sdxl_lora_ghibli = ( model_path("civitai/sdxl_10_lora/Studio Ghibli Style.safetensors"), "Studio Ghibli Style" )
    sdxl_lora_chalkboard = ( model_path("civitai/sdxl_10_lora/SDXL_ChalkBoardDrawing_LoRA_r8.safetensors"), "ChalkBoardDrawing" )

    depth_image_file = os.path.join(cwd, "../diffusion-server-files/input-depth.png")
    output_image = os.path.join(cwd, "../diffusion-server-files/sdxl_controlnet_test.png")

    server = xmlrpc.client.ServerProxy('http://localhost:1337')
    print(server.get_status())

    server.init(sdxl_model_mohawk, [sdxl_lora_offset_noise, sdxl_lora_scifistyle], [0.2, 0.6])

    seed = 123
    steps = 20
    guidance=7.5
    lora_scale = 1.0
    
    depth_image_png_bytes = xmlrpc.client.Binary(image_to_png_bytes(Image.open(depth_image_file)))
    result = server.generate_panorama(prompt3, negative_prompt, seed, steps, guidance, depth_image_png_bytes, lora_scale)
    image = image_from_png_bytes(result.data)
    image.save(output_image, "PNG")

    image.show()