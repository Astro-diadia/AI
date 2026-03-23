import numpy as np
import time

class Buffer:
    def __init__(self, speech_threshold=0.048, is_mic=True):
        self.speech_threshold = speech_threshold
        self.silence_time = 0.7
        self.buffer_stt = []  
        self.last_speech = None

        self.is_mic = is_mic
        self.prev_ratio = 0.0
        self.buffer_direction = []
        self.direction = "center"

    def process_chunk(self, chunk):
        chunk = chunk.astype(np.float32)

        if np.abs(chunk).mean() > self.speech_threshold:
            if not self.is_mic:
                direction_local = self.classify_direction(chunk)

                if direction_local is not None:
                    self.direction = direction_local
                    self.buffer_direction.clear()

            # self.buffer_stt.append(chunk.mean(axis=1))
            self.buffer_stt.append(chunk)

            self.last_speech = time.time()
            return None

        if self.last_speech is None:
            return None

        if time.time() - self.last_speech >= self.silence_time and len(self.buffer_stt) > 10:
            if not self.buffer_stt:
                return None

            audio = np.concatenate(self.buffer_stt)

            self.buffer_stt = []
            self.last_speech = None
            self.buffer_direction.clear()

            return {"audio": audio, "direction": self.direction}
        return None

    def classify_direction(self, chunk, threshold=0.1):
        self.buffer_direction.append(chunk)

        if len(self.buffer_direction) >= 8:
            big_chunk = np.concatenate(self.buffer_direction)

            left = np.abs(big_chunk[:, 0]).mean()
            right = np.abs(big_chunk[:, 1]).mean()

            ratio = (left - right) / (left + right + 1e-6)

            smoothed_ratio = 0.8 * self.prev_ratio + 0.2 * ratio

            self.prev_ratio = smoothed_ratio

            if smoothed_ratio > threshold:
                return "left"
            elif smoothed_ratio < -threshold:
                return "right"
            else:
                return "center"
        return None