# from faiss import IndexFlatIP
from faster_whisper import WhisperModel
# from speechbrain.pretrained import EncoderClassifier
from Stt.AudioCapture import AudioCapture
from Stt.Buffer import Buffer
import time
import queue
import threading
# import numpy as np

# py "D:\AI\Stt\Stt.py"

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

        self.system_buffer = Buffer(0.05, is_mic=False)
        self.mic_buffer = Buffer(0.015, is_mic=True)

        self.output_queue = queue.Queue()

        self.done = False
        self.worker = threading.Thread(
            target=self.stt_worker,
            daemon=True
        )
        self.worker.start()

    def stt_worker(self):
        user_text = ""
        system_text = ""

        while not self.done:
            chunk_mic = self.mic_buffer.process_chunk(self.audio.get_mic_audio())
            chunk_system = self.system_buffer.process_chunk(self.audio.get_system_audio())

            if chunk_mic is not None:
                user_text = wisper(chunk_mic["audio"]).strip()

            if chunk_system is not None:
                system_text = wisper(chunk_system["audio"]).strip()

            if (system_text != "" or user_text != ""):
                self.output_queue.queue.put({
                    "system_text": system_text,
                    "user_text": user_text,
                    "direction": chunk_system["direction"]
                    })
                user_text = ""
                system_text = ""

            time.sleep(0.01)

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

    def wisper(self, chunk):
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

    def get(self):
        return self.output_queue.queue.get()

    def stop(self):
        self.done = True
        slef.thread.join()
