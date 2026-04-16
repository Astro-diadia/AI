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

        self.mic_volume_processor = process_volume()
        self.system_volume_processor = process_volume()

        self.text_buffer = {
            "system_text": [],
            "mic_text": [],
        }

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

                self.mic_flush = mic_input["flush"]

                self.mic_volume = mic_input["volume"]

                mic_text = mic_input["text"]
                if mic_text is not None:
                    self.text_buffer["mic_text"].append(mic_text)
            except queue.Empty:
                pass

            try:
                system_input = self.stt.get_system()

                self.system_flush = system_input["flush"]

                self.system_volume = system_input["volume"]

                self.direction = system_input["direction"]
                
                system_text = system_input["text"]
                if system_text is not None:
                    self.text_buffer["system_text"].append(system_text)
            except queue.Empty:
                pass

            if self.system_flush or self.system_volume_processor.process_volume(self.system_volume):
                print("volume system")
                #callllm("systyem end, sumarise what user said")
 
            if self.mic_flush or self.mic_volume_processor.process_volume(self.mic_volume):
                print("volume mic")
                #callllm("user end, sumarise what systyem said")
            
            time.sleep(0.01)

    def build_prompt(self):
        history = self.short_mem.get()

        prompt = "Conversation history:\n"

        for message in history:
            prompt += f"{message['role']}: {message['content']}\n"

        if self.memory_query != "":
            memory = self.mid_mem.retrieve_similar(self.memory_query)
            prompt += "Relevant memory:\n"

            if len(memory) > 0:
                prompt += "\n".join(memory) + "\n"
            else:
                prompt += "\n".join(self.long_mem.retrieve_similar(self.memory_query)) + "\n"

            self.memory_query = ""

        self.text_buffer

        return prompt

    # def run_turn(self, user_input, system_input):
    #     self.short_mem.add(content=user_input, speaker_id="user1", role="user", mid_memory = self.mid_mem)   
    #     self.short_mem.add(content=system_input, speaker_id="system", role="system", mid_memory = self.mid_mem)   

    #     prompt = self.build_prompt(user_input, system_input)
    #     output = self.llm.generate(prompt)
    #     self.memory_query = output["what_to_searchy_in_memory"]

    #     self.short_mem.add(content=output["thought"], speaker_id="assistant", role="assistant", mid_memory = self.mid_mem)

    #     return output


class process_volume:
        def __init__(self):
            self.prev_volume = None
            self.noise_floor = None
            self.last_trigger_time = 0.0

        def process_volume(self, volume):
            now = time.time()

            if self.prev_volume is None:
                self.prev_volume = volume
                self.noise_floor = volume
                return False

            alpha_fast = 0.2
            smoothed = alpha_fast * volume + (1 - alpha_fast) * self.prev_volume

            if smoothed < self.noise_floor:
                alpha_slow = 0.05
            else:
                alpha_slow = 0.005

            self.noise_floor = alpha_slow * smoothed + (1 - alpha_slow) * self.noise_floor
            self.noise_floor = max(self.noise_floor, 1e-4)

            delta = smoothed - self.noise_floor

            trigger_threshold = max(0.005, self.noise_floor * 3)
            
            cooldown = 0.8

            if delta > trigger_threshold and (now - self.last_trigger_time) > cooldown:
                self.last_trigger_time = now
                self.prev_volume = smoothed
                return True

            self.prev_volume = smoothed
            return False

agent = Agent("LLMCore()", "ShortMemory()", "MidMemory()", "LongMemory()", Stt())
