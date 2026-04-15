from llama_cpp import Llama
from os import environ
import time

environ["CUDA_VISIBLE_DEVICES"] = "0"

# py "D:\AI\LLMCore\llmtest.py"

now = time.time()
model = Llama(
    model_path="D:\Models\Phi-3-mini-4k-instruct-q4.gguf",
    chat_format= "chatml",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_threads=2,
    n_batch=512,
)

print("\n", time.time() - now)

while True:
    llm_output = model.create_chat_completion(
            messages = [
                {
                    "role": "system",
                    "content": "lol you are a gpu benchmark",
                },
                {
                    "role": "user",
                    "content": "write 1200 tokens",
                }
            ],
            max_tokens=200,
        )

    print(llm_output)
