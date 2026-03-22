import numpy as np
from time import time

class Buffer:
    def __init__(self, speech_threshold=0.048, samplerate=16000, silence_time=0.4):
        self.speech_threshold = speech_threshold
        self.silence_time = silence_time
        self.buffer_stt = []  
        self.last_speech = None

    def process_chunk(self, chunk):
        chunk = chunk.astype(np.float32)

        if np.abs(chunk).mean() > self.speech_threshold:
            self.buffer_stt.append(chunk.mean(axis=1))

            self.last_speech = time()
            return None

        if self.last_speech is None:
            return None

        if time() - self.last_speech >= self.silence_time:

            if not self.buffer_stt:
                return None

            audio = np.concatenate(self.buffer_stt)

            self.buffer_stt = []
            self.last_speech = None

            return audio
        return None