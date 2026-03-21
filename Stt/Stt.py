# from faiss import IndexFlatIP
from faster_whisper import WhisperModel
# from speechbrain.pretrained import EncoderClassifier
from Stt.AudioCapture import AudioCapture
from Stt.Buffer import Buffer
from time import time
import queue
# import numpy as np

# py "D:\Stt\Stt.py"

class Stt:
    def __init__(self):
        self.stt_model = WhisperModel("small", device="cuda", compute_type="int8")
        # self.spk_model = EncoderClassifier.from_hparams(
        #     source="speechbrain/spkrec-ecapa-voxceleb",
        #     run_opts={"device": "cuda"}
        # )
 
        # self.dim = 192
        # self.index = IndexFlatIP(self.dim)
        # self.speaker_db = []

        self.audio = AudioCapture()
        self.audio.capture_mic()
        self.audio.capture_system()

        self.mic_buffer = Buffer()
        self.system_buffer = Buffer()

        self.output_queue = queue.Queue()

        self.prev_ratio = 1

        self.done = False
        self.worker = threading.Thread(
            target=self.stt_worker,
            daemon=True
        )
        self.worker.start()

    def stt_worker(self):
        while not self.done:
            chunk_system = self.audio.get_system_audio()
            chunk_mic = self.audio.get_mic_audio()

                        

            if time() - self.last_speech >= self.silence_time:
                

                mic_data = self.process_mic()
                if mic_data:
                    self.output_queue.put(mic_data)

                sys_data = self.process_system()
                if sys_data:
                    self.output_queue.put(sys_data)

    # def get_embedding(self, audio_file):
    #     emb = self.spk_model.encode_batch(audio_file)
    #     emb = emb.squeeze().cpu().numpy()
    #     emb = emb / np.linalg.norm(emb)
    #     return emb.astype("float32")

    # def classify_speaker(self, embedding, threshold=0.75):
    #     if self.index.ntotal == 0:
    #         self.index.add(np.array([embedding]))
    #         self.speaker_db.append("spk_0")
    #         return "spk_0"

    #     distances, indices = self.index.search(np.array([embedding]), 1)
    #     if distances[0][0] > threshold:
    #         return self.speaker_db[indices[0][0]]
    #     else:
    #         new_id = f"spk_{len(self.speaker_db)}"
    #         self.index.add(np.array([embedding]))
    #         self.speaker_db.append(new_id)
    #         return new_id

    def classify_direction(chunk, threshold=0.02, balance=1.2):
        left = np.abs(chunk[:, 0]).mean()
        right = np.abs(chunk[:, 1]).mean()

        ratio = (left - right) / (left + right + 1e-6)

        smoothed_ratio = 0.8 * self.prev_ratio + 0.2 * ratio

        self.prev_ratio = smoothed_ratio

        if ratio > balance:
            return "left"
        elif ratio < 1 / balance:
            return "right"
        else:
            return "center"

    def wisper(self, source, chunk):
        segments, _ = self.stt_model.transcribe(
            chunk,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            return None

        return text

    def process_system(self, chunk):
        text = self.wisper("system", chunk)

        direction = classify_direction(chunk)

        if text:
            return {
                "text": text,
                "direction": direction,
                "source": "system"
            }

        return None

    def process_mic(self, chunk):
        text = self.wisper("mic", chunk)

        if text:
            return {
                "text": text,
                "direction": "center", #remove?
                "source": "mic"
            }

        return None

    def stop(self):
        self.done = True
        slef.thread.join()
