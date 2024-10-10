import xmlrpc.client
import threading
import time


class SDXLClient:

    in_progress_ids = []

    def worker(self):
        worker_server = xmlrpc.client.ServerProxy('http://localhost:1337')
        while True:
            time.sleep(1)
            if len(self.in_progress_ids) > 0:
                completed_jobs = worker_server.list_completed_jobs()
                first_matching = next((x for x in self.in_progress_ids if x["id"] in completed_jobs), None)
                if first_matching is not None:
                    self.in_progress_ids.remove(first_matching)

                    id = first_matching["id"]
                    callback = first_matching["callback"]

                    print(f"SDXLClient : retrieving completed id {id}")

                    image_file = worker_server.get_image_file(id)

                    callback(id, image_file)

    def __init__(self):
        self.server = xmlrpc.client.ServerProxy('http://localhost:1337')
        threading.Thread(target=self.worker, daemon=True).start()

    def init(self, model_file:str, loras:list[tuple[str, str]] = [], lora_weights:list[float] = []) -> None:
        self.server.init(model_file, loras, lora_weights)
        self.in_progress_ids.clear()

    def generate_panorama(self, output_file:str, prompt:str, negative_prompt:str, seed:int, steps:int, prompt_guidance:float, depth_image_file:str, depth_image_influence:float, lora_overall_influence:float = 0) -> str:
        result_file = self.server.generate_panorama(output_file, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image_file, depth_image_influence, lora_overall_influence)
        return result_file

    def queue_panorama(self, output_file:str, callback, prompt:str, negative_prompt:str, seed:int, steps:int, prompt_guidance:float, depth_image_file:str, depth_image_influence:float, lora_overall_influence:float = 0) -> int:
        id = self.server.queue_panorama(output_file, prompt, negative_prompt, seed, steps, prompt_guidance, depth_image_file, depth_image_influence, lora_overall_influence)
        self.in_progress_ids.append({"id":id, "callback":callback})
        return id