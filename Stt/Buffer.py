import numpy as np
import time
import queue
import threading
import webrtcvad

# py "D:\AI\Stt\Buffer.py"

class Buffer:
    def __init__(self, volume_threshold=0.048, is_mic=True, get_audio_function=None):
        self.is_mic = is_mic
        self.volume_threshold = volume_threshold 
        self.get_audio_function = get_audio_function

        self.buffer = []
        self.buffer_len = 0

        self.silence_frame = 12
        self.silence_frame_short = 6
        self.silence_counter = 0
        self.vad = webrtcvad.Vad(2)
        self.frame_size = 320

        self.max_audio = 16000 * 2.6

        self.output_queue = queue.Queue(maxsize=15)

        self.prev_ratio = 0.0
        self.buffer_direction = []
        self.direction = "center"

        self.done = False
        self.worker = threading.Thread(
            target=self.buffer_main,
            daemon=True
        )
        self.worker.start()

    def buffer_main(self):
        while not self.done:
            try:
                block = self.get_audio_function()
                self.process_block(block)
            except queue.Empty:
                pass

            time.sleep(0.01)

    def process_block(self, block):
        volume = np.sqrt((block**2).mean())

        if not self.is_mic:
            self.classify_direction(block)

        # if volume < self.volume_threshold:
        #     self.silence_counter += 1

        if block.ndim == 2:
            block = block.mean(axis=1)

        if self.is_speech(block):
            self.buffer.append(block)
            self.buffer_len += 1024

        if self.silence_counter >= self.silence_frame:
            if self.buffer_len == 0:
                return

            self.silence_counter = 0
            self.buffer_len = 0

            audio = np.concatenate(self.buffer)

            self.buffer.clear()

            if not self.output_queue.full():
                data = {
                    "flush": False,
                    "audio": audio,
                    "volume": volume
                }

                if not self.is_mic:
                    data["direction"] = self.direction
                self.output_queue.put(data)

                return None
        
        if self.buffer_len >= self.max_audio and self.silence_counter >= self.silence_frame_short:
            self.silence_counter = 0
            self.buffer_len = 0

            audio = np.concatenate(self.buffer)
            
            if not self.output_queue.full():
                data = {
                    "flush": False,
                    "audio": audio,
                    "volume": volume
                }

                if not self.is_mic:
                    data["direction"] = self.direction

                self.output_queue.put(data)

                overlap = int(16000 * 0.2)

                self.buffer = [audio[-overlap:]]
                self.buffer_len = overlap

        return None

    def classify_direction(self, block, threshold=0.0):
        self.buffer_direction.append(block)

        if len(self.buffer_direction) >= 4:
            four_block = np.concatenate(self.buffer_direction)

            left = np.abs(four_block[:, 0]).mean()
            right = np.abs(four_block[:, 1]).mean()

            ratio = (left - right) / (left + right + 1e-6)

            self.prev_ratio = 0.8 * self.prev_ratio + 0.2 * ratio

            if self.prev_ratio > threshold:
                self.direction = "left"
            elif self.prev_ratio < -threshold:
                self.direction = "right"
            else:
                self.direction = "center"

            self.buffer_direction.clear()

        return None

    def is_speech(self, chunk_float32):
        chunk_int16 = (chunk_float32 * 32768).astype(np.int16)

        speech_detected = False

        for i in range(0, len(chunk_int16) - self.frame_size + 1, self.frame_size):
            frame = chunk_int16[i:i+self.frame_size]

            if self.vad.is_speech(frame.tobytes(), 16000):
                speech_detected = True

        if speech_detected:
            self.silence_counter = 0
        else:
            self.silence_counter += 1

        return speech_detected

    def get_audio(self):
        return self.output_queue.get(timeout=0.1)

    def stop(self):
        self.done = True
         