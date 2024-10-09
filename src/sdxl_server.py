from sdxl_engine import SDXL
from utils import LoraInfo, image_to_png_bytes

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
            
            @server.register_function(name='enqueue_with_depth')
            def enqueue_with_depth(prompt:str, negative_prompt:str, depth_image_file:str, lora_scale:float = 0) -> None:
                print("Server : enqueue_with_depth() called")
                if self.sdxl is None: return
                result_image = self.sdxl.generate_using_depth(prompt=prompt, negative_prompt=negative_prompt, lora_scale=lora_scale, depth_image_file=depth_image_file)
                return xmlrpc.client.Binary(image_to_png_bytes(result_image))
            
            print("Server : serving...")
            server.serve_forever()

if __name__ == "__main__":
    SDXLServer().start()