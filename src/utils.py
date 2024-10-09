import os, io
from timeit import default_timer as timer
from PIL import Image

def log_timing(prev_time:float, message:str) -> float:

    this_time = timer()
    print("********")
    print(f"{message} @ {this_time} (last step took {this_time - prev_time})")
    print("********")
    
    return this_time

def model_path(path_under_models_folder:str) -> str:
    models_base = "../../../models/"
    return os.path.join(models_base, path_under_models_folder)

def image_from_png_bytes(data:bytes) -> Image:
    b =  io.BytesIO(data)
    return Image.open(b)

def image_to_png_bytes(img:Image) -> bytes:
    b = io.BytesIO()
    img.save(b, 'PNG')
    return b.getvalue()

class LoraInfo:
    def __init__(self, model:str, key:str) -> None:
        self.model = model
        self.key = key
