import os
from timeit import default_timer as timer

def log_timing(prev_time:float, message:str) -> float:

    this_time = timer()
    print("********")
    print(f"{message} @ {this_time} (last step took {this_time - prev_time})")
    print("********")
    
    return this_time

def model_path(path_under_models_folder:str) -> str:
    models_base = "../../../models/"
    return os.path.join(models_base, path_under_models_folder)


class LoraInfo:
    def __init__(self, model:str, key:str) -> None:
        self.model = model
        self.key = key
