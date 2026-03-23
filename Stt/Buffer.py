import numpy as np
import time
import queue
import threading
from Stt.AudioCapture import AudioCapture

class Buffer:
    def __init__(self, speech_threshold=0.048):
        self.speech_threshold = speech_threshold
        self.silence_time = 0.7
        self.buffer_stt_mic = []
        self.buffer_stt_sys = []
        self.last_speech_mic = None
        self.last_speech_sys = None

        self.AudioCapture = AudioCapture()
        self.AudioCapture.capture_mic()
        self.AudioCapture.capture_system()

        self.min_audio = 16000 * 1.6

        self.output_queue_system = queue.Queue(maxsize=60)
        self.output_queue_mic = queue.Queue(maxsize=60)

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
                chunk = self.AudioCapture.get_mic_audio()
                self.process_mic(chunk=chunk)
            except queue.Empty:
                pass

            try:
                chunk = self.AudioCapture.get_system_audio()
                self.process_system(chunk=chunk)
            except queue.Empty:
                pass

    def process_system(self, chunk):
        if np.sqrt((chunk**2).mean()) > self.speech_threshold:
            direction_local = self.classify_direction(chunk)

            if direction_local is not None:
                self.direction = direction_local
                self.buffer_direction.clear()

            self.buffer_stt_sys.append(chunk.mean(axis=1))

            self.last_speech_sys = time.time()
            return None

        if self.last_speech_sys is None:
            return None

        if time.time() - self.last_speech_sys >= self.silence_time:
            if not self.buffer_stt_sys:
                return None

            audio = np.concatenate(self.buffer_stt_sys)

            if len(audio) < self.min_audio:
                return None

            self.buffer_stt_sys = []
            self.last_speech_sys = None
            self.buffer_direction.clear()

            if not self.output_queue_system.full():
                self.output_queue_system.put({
                    "audio": audio,
                    "direction": self.direction
                })

        return None

    def process_mic(self, chunk):
        if np.sqrt((chunk**2).mean()) > self.speech_threshold:

            self.buffer_stt_mic.append(chunk)

            self.last_speech_mic = time.time()
            return None

        if self.last_speech_mic is None:
            return None

        if time.time() - self.last_speech_mic >= self.silence_time:
            if not self.buffer_stt_mic:
                return None

            audio = np.concatenate(self.buffer_stt_mic)

            if len(audio) < self.min_audio:
                return None            

            self.buffer_stt_mic = []
            self.last_speech_mic = None

            if not self.output_queue_mic.full():
                self.output_queue_mic.put({
                    "audio": audio,
                    "direction": "center"
                })

        return None

    def classify_direction(self, chunk, threshold=0.1):
        self.buffer_direction.append(chunk)

        if len(self.buffer_direction) >= 4:
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

    def get_mic_audio(self):
        return self.output_queue_mic.get(timeout=0.1)

    def get_system_audio(self):
        return self.output_queue_system.get(timeout=0.1)

    def stop(self):
        self.done = True