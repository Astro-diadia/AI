import subprocess
import simpleaudio as sa
from queue import Queue, put, get, task_done
from threading import Thread, start

# py "D:\Tts\Tts.py"

class Tts:
    def __init__(self, model):
        self.model = model
        self.tts_queue = Queue()
        self.is_playing = False
        self.thread = Thread(target=self.tts_worker, daemon=True)
        self.thread.start()

    def speak(self, text):
        if text:
            self.tts_queue.put((text))

    def tts_worker(self):
        while True:
            text = self.tts_queue.get()
            try:
                self.is_playing = True
                text = text.replace("\x00", "").encode("utf-8", errors="ignore").decode("utf-8")

                wav_file = "D:\\eng-tts.wav"

                subprocess.run(
                    ["piper", "--model", self.model, "--output_file", "D:\\eng-tts.wav"],
                    input=text,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                sa.WaveObject.from_wave_file(wav_file).play().wait_done()
                # if (input("Press Enter to continue...") == "exit"):
                #     break
            finally:
                self.is_playing = False
                self.tts_queue.task_done()
