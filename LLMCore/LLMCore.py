from llama_cpp import Llama, LlamaGrammar
from json import loads
import threading
import queue
import time
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "0"

# py "D:\AI\LLMCore\LLMCore.py"

class LLMCore:
    def __init__(
            self,
            max_tokens=120,
            content="you are an ai assistent"
            ):
        lol = time.time()
        self.model = Llama(
            model_path="D:\Models\Phi-3-mini-4k-instruct-q4.gguf",
            chat_format= "chatml",
            n_gpu_layers=-1,
            n_ctx=2048,
            n_threads=4,
            n_batch=512,
            use_mmap=False,
            use_mlock=True,
            verbose=False,
            cache=True,
        )

        print("\n", time.time() - lol)

        self.max_tokens = max_tokens
        self.content = content

        self.buffer = ""

        self.emit_time = 0.5
        
        self.queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=30)

        self.done = False
        self.worker = threading.Thread(
            target=self.llm_worker,
            daemon=True
        )
        self.worker.start()

    def llm_worker(self):
        while not self.done:
            try:
                prompt = self.queue.get(timeout=0.1)
                if prompt is not None:
                    buffer = []
                    last_emit = time.time()

                    llm_output = self.model.create_chat_completion(
                            messages = [
                                {
                                    "role": "system",
                                    "content": self.content,
                                },
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            max_tokens=self.max_tokens,
                            stream=True
                        )

                    for token in llm_output:
                        if self.done:
                            break

                        if not token:
                            continue
                        
                        buffer.append(token['choices'][0]['delta'].get('content', ''))

                        if self.should_emit(buffer, last_emit):
                            text = ''.join(buffer).strip()
                            self.output_queue.put(text)
                            buffer.clear()
                            last_emit = time.time()

                    if buffer:
                        self.output_queue.put(''.join(buffer).strip())

                    self.output_queue.put(None)
            except queue.Empty:
                pass

    def should_emit(self, buffer, last_emit):
        if not buffer:
            return False

        if buffer[len(buffer) - 1] in ".!?" and buffer[-1] == " ":
            # print("encountered .!?")
            return True

        if buffer[-1] in ",;:" and len(buffer) > 30:
            # print("encountered ,;:")
            return True

        if len(buffer) > 20 and time.time() - last_emit > self.emit_time:
            # print("last_emit")
            return True

        if len(buffer) > 80:
            # print("buffer 80")
            return True

        return False           

    def generate(self, prompt):
        if prompt is None:
            return

        self.queue.put(prompt)

        while True:
            try:
                llm_output = self.output_queue.get(timeout=0.1)
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
