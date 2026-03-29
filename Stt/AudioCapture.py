import sounddevice as sd
import queue
# import numpy as np
# import wave

# py "D:\AI\Stt\AudioCapture.py"

class AudioCapture:
    def __init__(self, samplerate=16000, blocksize=1024):
        self.samplerate = samplerate
        self.blocksize = blocksize

        self.output_queue_mic = queue.Queue(maxsize=40)
        self.output_queue_system = queue.Queue(maxsize=40)

        self.mic_stream = None
        self.sys_stream = None

    def _mic_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if not self.output_queue_mic.full():
            self.output_queue_mic.put(indata.copy())

    def _sys_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if not self.output_queue_system.full():
            self.output_queue_system.put(indata.copy())

    def capture_mic(self, device=1):
        self.mic_stream = sd.InputStream(
            device=device,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            callback=self._mic_callback
        )

        self.mic_stream.start()

    def capture_system(self, device=5):
        self.sys_stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=device,
            channels=2,
            callback=self._sys_callback,
        )

        self.sys_stream.start()

    def get_mic_audio(self):
        return self.output_queue_mic.get(timeout=0.1)

    def get_system_audio(self):
        return self.output_queue_system.get(timeout=0.1)

    def stop(self):
        if self.mic_stream:
            self.mic_stream.stop()
        if self.sys_stream:
            self.sys_stream.stop()

# print(sd.query_devices(device=5)['default_samplerate'])
# print(sd.query_devices())

# audio = AudioCapture()
# audio.capture_system(5) #1-mic 5-system 

# wav_file = wave.open("D:\\sys_output.wav", "wb")
# wav_file.setnchannels(2)
# wav_file.setsampwidth(2)
# wav_file.setframerate(16000)

# try:
#     print("Recording... Press Ctrl+C to stop.")
#     while True:

#         chunk = audio.get_system_audio()
        
#         # chunk = chunk[:, :2]

#         # print(chunk.min(), "\n", chunk.max(), "\n")

#         chunk_int16 = (chunk * 32767).astype(np.int16)

#         wav_file.writeframes(chunk_int16.tobytes())

# except KeyboardInterrupt:
#     pass

# wav_file.close()
# audio.stop()