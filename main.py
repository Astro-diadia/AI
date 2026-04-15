from LLMCore.LLMCore import LLMCore
from MemControl.LongMemory import LongMemory
from MemControl.ShortMemory import ShortMemory
from MemControl.MidMemory import MidMemory
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

        self.volume_default = None

        self.text_buffer = {}
        self.flush = False
        self.direction = "center"

        self.done = False
        self.worker = threading.Thread(
            target=self.main_cicle,
            daemon=False
        )
        self.worker.start()

    def main_cicle(self):
        while not self.done:
            try:
                mic_input = self.stt.get_mic()
                mic_volume = mic_input["volume"]
                print("mic_volume", mic_volume)
                self.flush = mic_input["flush"]

                mic_text = mic_input["text"]

                print(mic_text)
            except queue.Empty:
                pass

            try:
                system_input = self.stt.get_system()
                system_volume = system_input["volume"]
                print("system_volume", system_volume)
                self.direction = system_input["direction"]
                self.flush = system_input["flush"]
                
                system_text = system_input["text"]
                print(system_text)
            except queue.Empty:
                pass

            if self.flush:
                pass
                #call llm
            
            time.sleep(0.01)

    def process_volume(self, volume):
        if self.volume_default is None:
            self.volume_default = volume
            
            return None

        if volume > self.volume_default:
            pass
            # call llm

        self.volume_default = (self.volume_default + volume) / 2

        


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
    print("lol you fucked up", "\n", 'py "D:\AI\main.py"')
