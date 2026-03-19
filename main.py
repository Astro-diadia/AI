# from LLMCore.LLMCore import LLMCore
# from MemControl.LongMemory import LongMemory
# from MemControl.ShortMemory import ShortMemory
# from MemControl.MidMemory import MidMemory
from Stt.Stt import Stt
# from Tts.Tts import Tts
from os import environ
from time import time, sleep

environ["CUDA_VISIBLE_DEVICES"] = "1"

# py "D:\main.py"

class Agent:
    def __init__(self, llm, short_mem, mid_mem, long_mem, stt):
        self.llm = llm
        self.short_mem = short_mem
        self.mid_mem = mid_mem
        self.long_mem = long_mem
        self.stt = stt
        self.memory_query = ""

    def process_mic(self):
        return self.stt.process_mic()

    def process_system(self):
        return self.stt.process_system()

    def build_prompt(self, user_input, system_input):
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

        prompt += f"\nCurrent user message:\n {user_input}\n"
        prompt += f"\nCurrent system sounds:\n {system_input}\n"

        return prompt

    def run_turn(self, user_input, system_input):
        self.short_mem.add(content=user_input, speaker_id="user1", role="user", mid_memory = self.mid_mem)   
        self.short_mem.add(content=system_input, speaker_id="system", role="system", mid_memory = self.mid_mem)   

        prompt = self.build_prompt(user_input, system_input)
        output = self.llm.generate(prompt)
        self.memory_query = output["what_to_searchy_in_memory"]

        self.short_mem.add(content=output["thought"], speaker_id="assistant", role="assistant", mid_memory = self.mid_mem)

        return output

agent = Agent(LLMCore(), ShortMemory(), MidMemory(), LongMemory(), Stt())
print("start")

timer = None
accumulated_input = [] 
timeConst = 3

while True:
    user_input = agent.process_mic()
    system_input = agent.process_system()

    if user_input is not None || system_input is not None:
        print("You:", user_input, "\n")
        print("AI:", agent.run_turn(user_input, system_input)["speak"], "\n")

    sleep(0.1)
