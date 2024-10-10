from sdxl_engine import SDXL
from utils import LoraInfo, image_from_png_bytes, image_to_png_bytes

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client

import threading
import queue
import time

class SDXLServer:

    server:SimpleXMLRPCServer = None
    sdxl:SDXL = None
    status:str = 'needs-init'
    current_id:int = 0

    def __init__(self):
        self.sdxl = SDXL()

    def start(self):

        # The worker thread will act on queued jobs sequentially
        completed_jobs = []
        q_in = queue.Queue()
        def worker():
            while True:
                if q_in.empty():
                    time.sleep(1)
                else:
                    settings = q_in.get()
                    id = settings["id"]
                    output_file = settings["output_file"]
                    print(f"Processing request {id}")
                    image = self.sdxl.generate_panorama(prompt=settings["prompt"], negative_prompt=settings["negative_prompt"], seed=settings["seed"], steps=settings["steps"], prompt_guidance=settings["prompt_guidance"], depth_image=settings["depth_image"], depth_image_influence=settings["depth_image_influence"], lora_overall_influence=settings["lora_overall_influence"])
                    image.save(output_file)
                    q_in.task_done()
                    completed_jobs.append({"id":id, "image_file":output_file})
                    
        # Start the worker thread.
        threading.Thread(target=worker, daemon=True).start()

        # Create the xmlrpc server and register functions
        with SimpleXMLRPCServer(('localhost', 1337), requestHandler=SimpleXMLRPCRequestHandler) as server:

            server.register_introspection_functions()

            @server.register_function(name='get_status')
            def get_status() -> str:
                print("Server : get_status() called")
                return self.status
            
            @server.register_function(name='init')
            def init(model_file:str, loras:list[tuple[str, str]] = [], lora_weights:list[float] = []) -> None:
                print("Server : init() called")
                loras = [LoraInfo(model, key) for model, key in loras]
                self.sdxl.init(model_file=model_file, loras=loras, lora_weights=lora_weights)
                self.status = 'ready'
                return self.status
            
            @server.register_function(name='generate_panorama')
            def generate_panorama(output_file:str, prompt:str, negative_prompt:str, seed:int, steps:int, prompt_guidance:float, depth_image_png_bytes:xmlrpc.client.Binary, depth_image_influence:float, lora_overall_influence:float = 0) -> str:
                print("Server : generate_panorama() called")
                if self.sdxl is None: return
                depth_image = image_from_png_bytes(depth_image_png_bytes.data)
                result_image = self.sdxl.generate_panorama(prompt=prompt, negative_prompt=negative_prompt, seed=seed, steps=steps, prompt_guidance=prompt_guidance, depth_image=depth_image, depth_image_influence=depth_image_influence, lora_overall_influence=lora_overall_influence)
                result_image.save(output_file)
                return output_file
            
            @server.register_function(name='queue_panorama')
            def queue_panorama(output_file:str, prompt:str, negative_prompt:str, seed:int, steps:int, prompt_guidance:float, depth_image_png_bytes:xmlrpc.client.Binary, depth_image_influence:float, lora_overall_influence:float = 0) -> int:
                print("Server : queue_panorama() called")
                if self.sdxl is None: return
                depth_image = image_from_png_bytes(depth_image_png_bytes.data)
                
                settings = {
                    "id":self.current_id,
                    "output_file":output_file,
                    "prompt":prompt, 
                    "negative_prompt":negative_prompt, 
                    "seed":seed, 
                    "steps":steps, 
                    "prompt_guidance":prompt_guidance, 
                    "depth_image":depth_image, 
                    "depth_image_influence":depth_image_influence, 
                    "lora_overall_influence":lora_overall_influence
                }
                self.current_id += 1
                q_in.put(settings)

                return settings["id"]
            
            @server.register_function(name='list_completed_jobs')
            def list_completed_jobs() -> list[int]:
                print("Server : list_completed_jobs() called")
                return list([item["id"] for item in completed_jobs])
            
            @server.register_function(name='get_image_file')
            def get_image_file(id:int) -> str | None:
                item_with_id = next((x for x in completed_jobs if x["id"] == id), None)
                if item_with_id is not None:
                    completed_jobs.remove(item_with_id)
                    return item_with_id["image_file"]
                else:
                    return None

            print("Server : serving...")
            server.serve_forever()

if __name__ == "__main__":
    SDXLServer().start()
