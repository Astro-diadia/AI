from llama_cpp import Llama
from json import loads
import threading
import queue
from time import sleep, time
from os import environ

# environ["CUDA_VISIBLE_DEVICES"] = "0"

# py "D:\AI\LLMCore\LLMCore.py"

class LLMCore:
    def __init__(self):
        self.model = Llama(
            model_path="D:\Models\Phi-3-mini-4k-instruct-q4.gguf",
            # chat_format= "mistral-instruct",
            chat_format= "chatml",
            n_gpu_layers=-1,
            n_ctx=2048,
            n_threads=2, #TODO test 2 vs 4
            n_batch=512,
            use_mmap=True,
            use_mlock=True,
            verbose=False
        )

        self.buffer = ""
        
        self.queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.done = False
        self.worker = threading.Thread(
            target=self.llm_worker,
            daemon=True
        )
        self.worker.start()

    def llm_worker(self):
        while not self.done:
            prompt = self.queue.get()
            if prompt is None:
                break

            buffer = ""
            last_emit = time()

            llm_output = self.model.create_chat_completion(
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
                    max_tokens=100,
                    stream=True
                )

            for token in llm_output:
                if self.done:
                    break

                if not token:
                    continue
                
                buffer += token['choices'][0]['delta'].get('content', '')

                if self.should_emit(buffer, last_emit):
                    self.output_queue.put(buffer.strip())
                    buffer = ""
                    last_emit = time()

            if buffer:
                self.output_queue.put(buffer.strip())

            self.output_queue.put(None)

            self.queue.task_done()

    def should_emit(self, buffer, last_emit):
        if not buffer:
            return False

        if buffer[-1] in ".!?":
            print("encountered .!?")
            return True

        if buffer[-1] in ",;:" and len(buffer) > 30:
            print("encountered ,;:")
            return True

        if len(buffer) > 20 and time() - last_emit > 0.4:
            print("last_emit")
            return True

        if len(buffer) > 80:
            print("buffer 60")
            return True

        return False           

    def generate(self, prompt):
        self.queue.put(prompt)

        while True:
            try:
                llm_output = self.output_queue.get(timeout=1)
            except queue.Empty:
                continue

            if llm_output is None:
                break

            yield llm_output

    def stop(self):
        self.queue.clear()
        self.output_queue.clear()
        self.done = True
        self.worker.join()
