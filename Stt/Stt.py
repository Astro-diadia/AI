# from faiss import IndexFlatIP
from faster_whisper import WhisperModel
# from speechbrain.pretrained import EncoderClassifier
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

        self.mic_buffer = Buffer(0.048)
        self.system_buffer = Buffer(0.02)

        self.output_queue_system = queue.Queue(maxsize=15)
        self.output_queue_mic = queue.Queue(maxsize=15)

        self.done = False
        self.worker = threading.Thread(
            target=self.stt_worker,
            daemon=True
        )
        self.worker.start()

    def stt_worker(self):
        mic_text = None
        system_text = None

        while not self.done:
            try:
                mic_audio = self.mic_buffer.get_mic_audio()

                mic_text = self.whisper(mic_audio)

                if mic_text is not None:
                    if not self.output_queue_mic.full():
                        self.output_queue_mic.put(mic_text)
                        mic_text = None
            except queue.Empty:
                pass

            try:
                system_audio = self.system_buffer.get_system_audio()

                system_text = self.whisper(system_audio["audio"])

                if system_text is not None:
                    if not self.output_queue_system.full():
                        self.output_queue_system.put({
                            "text": system_text,
                            "direction": system_audio["direction"] or "unknown"
                            })
                        system_text = None                 
            except queue.Empty:
                pass

            time.sleep(0.01)

    def whisper(self, audio):
        segments, _ = self.stt_model.transcribe(
            audio,
            beam_size=4,
            vad_filter=False,
            condition_on_previous_text=False
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            return None

        return text

    def get_system_text(self):
        return self.output_queue_system.get(timeout=0.1)

    def get_mic_text(self):
        return self.output_queue_mic.get(timeout=0.1)

    def stop(self):
        self.done = True
        self.thread.join()

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
