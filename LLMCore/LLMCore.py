from llama_cpp import Llama, LlamaGrammar
from json import loads
import threading
import queue
import time
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
            n_threads=2,
            n_batch=1024, #512 original 80tps
            use_mmap=True,
            use_mlock=True,
            verbose=False,
            cache=True,
            # cache_prompt=True
        )

        root ::= "{" ws "\"tts\"" ws ":" ws string ws "," ws "\"tool_calls\"" ws ":" ws tool_array ws "}"

        tool_array ::= "[" ws (tool_call (ws "," ws tool_call)*)? ws "]"

        tool_call ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"arguments\"" ws ":" ws simple_object ws "}"

        simple_object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
        pair ::= string ws ":" ws simple_value

        simple_value ::= string | number | "true" | "false" | "null"

        string ::= "\"" chars "\""
        chars ::= "" | char chars
        char ::= [^"\\] | "\\" ["\\/bfnrt]

        number ::= "-"? [0-9]+ ("." [0-9]+)?

        ws ::= [ \t\n\r]*

        self.grammar = LlamaGrammar.from_string(gbnf_string)

        self.buffer = ""

        self.last_emit = 0.4
        
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
                    # grammar=self.grammar,
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

        if len(buffer) > 20 and time() - last_emit > self.last_emit:
            print("last_emit")
            return True

        if len(buffer) > 80:
            print("buffer 80")
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
