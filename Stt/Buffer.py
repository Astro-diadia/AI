import numpy as np
from time import time

class Buffer:
    def __init__(self, speech_threshold=0.048, samplerate=16000, silence_time=0.8):
        self.speech_threshold = speech_threshold
        self.silence_time = silence_time
        self.buffer = []  
        self.last_speech = None

    def process_block(self, block):
        block = block.mean(axis=1).astype(np.float32)

        if np.abs(block).mean() > self.speech_threshold:
            self.buffer.append(block)
            self.last_speech = time()
            return None

        if self.last_speech is None:
            return None

        if time() - self.last_speech >= self.silence_time:
            if not self.buffer:
                return None

            audio = np.concatenate(self.buffer)

            self.buffer = []
            self.last_speech = None
            return audio

        return None