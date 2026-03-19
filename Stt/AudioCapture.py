import sounddevice as sd
import queue
# import numpy as np
# import wave

# py "D:\Stt\AudioCapture.py"

class AudioCapture:
    def __init__(self, samplerate=16000, blocksize=1024):
        self.samplerate = samplerate
        self.blocksize = blocksize

        self.mic_queue = queue.Queue()
        self.sys_queue = queue.Queue()

        self.mic_stream = None
        self.sys_stream = None


    def _mic_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.mic_queue.put(indata.copy())


    def _sys_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.sys_queue.put(indata.copy())


    def capture_mic(self, device=1):
        self.mic_stream = sd.InputStream(
            device=device,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            callback=self._mic_callback
        )

        self.mic_stream.start()


    def capture_system(self, device=2):
        self.sys_stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=device,
            channels=2,
            callback=self._sys_callback,
        )

        self.sys_stream.start()


    def get_mic_audio(self):
        return self.mic_queue.get()


    def get_system_audio(self):
        return self.sys_queue.get()


    def stop(self):
        if self.mic_stream:
            self.mic_stream.stop()
        if self.sys_stream:
            self.sys_stream.stop()

# print(sd.query_devices())

# audio = AudioCapture()
# audio.capture_system(2)

# wav_file = wave.open("D:\\sys_output.wav", "wb")
# wav_file.setnchannels(2)
# wav_file.setsampwidth(2)
# wav_file.setframerate(16000)

# try:
#     print("Recording... Press Ctrl+C to stop.")
#     while True:

#         chunk = audio.get_system_audio()

#         chunk_int16 = (chunk * 32767).astype(np.int16)

#         wav_file.writeframes(chunk_int16.tobytes())

# except KeyboardInterrupt:
#     pass

# wav_file.close()
# audio.stop()