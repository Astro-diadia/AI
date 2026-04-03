# from LLMCore.LLMCore import LLMCore
# from MemControl.LongMemory import LongMemory
# from MemControl.ShortMemory import ShortMemory
# from MemControl.MidMemory import MidMemory
from Stt.Stt import Stt
# from Tts.Tts2 import Tts
from os import environ
import time
import queue
import threading

environ["CUDA_VISIBLE_DEVICES"] = "1"

# py "D:\AI\main.py"

class Agent:
    def __init__(self, llm, short_mem, mid_mem, long_mem, stt):
        self.llm = llm
        self.short_mem = short_mem
        self.mid_mem = mid_mem
        self.long_mem = long_mem
        self.stt = stt
        self.memory_query = ""

        self.done = False
        self.worker = threading.Thread(
            target=self.main_cicle,
            daemon=False
        )
        self.worker.start()

    def main_cicle(self):
        print("start")
        while not self.done:
            try:
                sys_text = self.get_system_text()
                print(sys_text["text"], "\n")
                print(sys_text["direction"], "\n")
            except queue.Empty:
                pass

            try:
                mic_text = self.get_mic_text()
                print(mic_text, "\n")
            except queue.Empty:
                pass

    def get_system_text(self):
        return self.stt.get_system_text()

    def get_mic_text(self):
        return self.stt.get_mic_text()

    # def build_prompt(self, user_input, system_input):
    #     history = self.short_mem.get()

    #     prompt = "Conversation history:\n"

    #     for message in history:
    #         prompt += f"{message['role']}: {message['content']}\n"

    #     if self.memory_query != "":
    #         memory = self.mid_mem.retrieve_similar(self.memory_query)
    #         prompt += "Relevant memory:\n"

    #         if len(memory) > 0:
    #             prompt += "\n".join(memory) + "\n"
    #         else:
    #             prompt += "\n".join(self.long_mem.retrieve_similar(self.memory_query)) + "\n"

    #         self.memory_query = ""

    #     prompt += f"\nCurrent user message:\n {user_input}\n"
    #     prompt += f"\nCurrent system sounds:\n {system_input}\n"

    #     return prompt

    # def run_turn(self, user_input, system_input):
    #     self.short_mem.add(content=user_input, speaker_id="user1", role="user", mid_memory = self.mid_mem)   
    #     self.short_mem.add(content=system_input, speaker_id="system", role="system", mid_memory = self.mid_mem)   

    #     prompt = self.build_prompt(user_input, system_input)
    #     output = self.llm.generate(prompt)
    #     self.memory_query = output["what_to_searchy_in_memory"]

    #     self.short_mem.add(content=output["thought"], speaker_id="assistant", role="assistant", mid_memory = self.mid_mem)

    #     return output


try:
    agent = Agent(LLMCore(), ShortMemory(), MidMemory(), LongMemory(), Stt())
except:
    printf("lol you fucked up", "\n", 'py "D:\AI\main.py"')
