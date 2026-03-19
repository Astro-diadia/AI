from llama_cpp import Llama
from json import loads
import threading
import queue
from time import sleep
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "0"

# py "D:\AI\LLMCore\LLMCore.py"

class LLMCore:
    def __init__(self):
        self.model = Llama(
            model_path="D:\Models\Phi-3-mini-4k-instruct-q4.gguf",
            # chat_format= "mistral-instruct",
            chat_format= "chatml",
            n_gpu_layers=-1,
            n_ctx=2048,
            n_threads=4,
            n_batch=512,
            use_mmap=True,
            use_mlock=True,
            verbose=False,
            stream=True
        )

        self.buffer = []
        
        self.queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.done = False
        self.worker = threading.Thread(
            target=self._llm_worker,
            daemon=True
        )
        self.worker.start()

    def _llm_worker(self):
        while not self.done:
            prompt = self.queue.get()

            if prompt is None:
                break

            for llm_output in self.model.create_chat_completion(
                    messages = [
                        {
                            "role": "system",
                            "content": "you are an ai assistant",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_tokens=100
                )['choices'][0]['message']['content']:
                self.buffer.append(llm_output)

            self.queue.task_done()

            sleep(0.1)
        self.done = False

    def llm_buffer(self):


    def generate(self, prompt):
        self.queue.put(prompt)

        llm_output = self.output_queue.get()

        return llm_output

    def get(self):
        return self.output_queue.get()

    def stop(self):
        self.queue.clear()
        self.output_queue.clear()
        self.done = True
        self.worker.join()

LLMCore = LLMCore()
while True:
    print("llm said: ", LLMCore.generate("hello, what is the role of an orchestrator in an ai app"), "\n")