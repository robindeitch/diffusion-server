from sdxl_engine import SDXL
from utils import LoraInfo, image_from_png_bytes, image_to_png_bytes

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client

class SDXLServer:

    server:SimpleXMLRPCServer = None
    sdxl:SDXL = None
    status:str = 'needs-init'

    def start(self):
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
                self.sdxl = SDXL(model_file=model_file, loras=loras, lora_weights=lora_weights)
                self.status = 'ready'
                return self.status
            
            @server.register_function(name='generate_panorama')
            def generate_panorama(prompt:str, negative_prompt:str, seed:int, steps:int, prompt_guidance:float, depth_image_png_bytes:xmlrpc.client.Binary, depth_image_influence:float, lora_overall_influence:float = 0) -> None:
                print("Server : generate_panorama() called")
                if self.sdxl is None: return
                depth_image = image_from_png_bytes(depth_image_png_bytes.data)
                result_image = self.sdxl.generate_panorama(prompt=prompt, negative_prompt=negative_prompt, seed=seed, steps=steps, prompt_guidance=prompt_guidance, depth_image=depth_image, depth_image_influence=depth_image_influence, lora_overall_influence=lora_overall_influence)
                return xmlrpc.client.Binary(image_to_png_bytes(result_image))
            
            print("Server : serving...")
            server.serve_forever()

if __name__ == "__main__":
    SDXLServer().start()
