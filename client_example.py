import os
import xmlrpc.client

cwd = os.getcwd()
models_base = os.path.join(cwd, "../../models/")

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

    def model_path(path_under_models_folder:str) -> str:
        return os.path.join(models_base, path_under_models_folder)

    sdxl_model_mohawk = model_path("civitai/sdxl_10/MOHAWK_v20.safetensors")
    sdxl_model_cardos = model_path("civitai/sdxl_10/cardosXL_v10.safetensors")

    sdxl_lora_scifistyle = (model_path("civitai/sdxl_10_lora/scifi_buildings_sdxl_lora-scifistyle-cinematic scifi.safetensors"), "#scifistyle, cinematic scifi")
    sdxl_lora_moebius = ( model_path("civitai/sdxl_10_lora/Moebius comic book_SDXL.safetensors"), "FRESHIDESH Moebius comic book" )
    sdxl_lora_eboy = ( model_path("civitai/sdxl_10_lora/Eboy_Pixelart_Style_XL.safetensors"), "eboy style" )
    sdxl_lora_anime = ( model_path("civitai/sdxl_10_lora/Anime_Sketch_SDXL.safetensors"), "(Pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch), unfinished, hatching (texture)" )
    sdxl_lora_woodcut = ( model_path("civitai/sdxl_10_lora/Landscape printsSDXL.safetensors"), "Landscape woodcut prints" )
    sdxl_lora_ghibli = ( model_path("civitai/sdxl_10_lora/Studio Ghibli Style.safetensors"), "Studio Ghibli Style" )
    sdxl_lora_chalkboard = ( model_path("civitai/sdxl_10_lora/SDXL_ChalkBoardDrawing_LoRA_r8.safetensors"), "ChalkBoardDrawing" )

    depth_image = os.path.join(cwd, "../diffusion-server-files/input-depth.png")
    output_image = os.path.join(cwd, "../diffusion-server-files/sdxl_controlnet_test.png")

    server = xmlrpc.client.ServerProxy('http://localhost:1337')
    print(server.get_status())

    server.init(sdxl_model_mohawk, [sdxl_lora_anime, sdxl_lora_moebius], [0.6, 0.9])
    print(server.get_status())
    
    server.enqueue_with_depth(prompt3, negative_prompt, depth_image, output_image, 1.0)