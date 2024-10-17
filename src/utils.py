import itertools
import math
import os, io
from timeit import default_timer as timer
from PIL import Image
import torch

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


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols

@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    dims = len(tile)
    output = torch.empty([samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b+1]

        # handle entire input fitting in a single tile
        if all(s.shape[d+2] <= tile[d] for d in range(dims)):
            output[b:b+1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        out = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)

        positions = [range(0, s.shape[d+2], tile[d] - overlap) if s.shape[d+2] > tile[d] else [0] for d in range(dims)]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)

            for t in range(feather):
                for d in range(2, dims + 2):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)
            o_d.add_(mask)

            if pbar is not None:
                pbar.update(1)

        output[b:b+1] = out/out_div
    return output

def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device, pbar)