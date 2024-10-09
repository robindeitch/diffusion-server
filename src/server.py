from engine_sdxl import SDXL
from utils import model_path, LoraInfo

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

    sdxl = SDXL(model_file=sdxl_model_mohawk, loras=[sdxl_lora_anime, sdxl_lora_moebius], lora_weights=[0.6, 0.9])
    sdxl.generate_using_depth(prompt=prompt3, negative_prompt=negative_prompt, lora_scale=1.0, depth_image_file=depth_image, output_image_file=img('test'))
