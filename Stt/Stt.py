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
        self.stt_model = WhisperModel("medium", device="cuda", compute_type="int8")
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

        self.system_buffer = Buffer(0.048, is_mic=False)
        self.mic_buffer = Buffer(0.015, is_mic=True)

        self.output_queue_system = queue.Queue()
        self.output_queue_mic = queue.Queue()

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

            if chunk_system is not None:
                system_text = self.whisper(chunk_system["audio"])
                if system_text:
                    self.output_queue_system.put({
                        "text": system_text,
                        "direction": chunk_system["direction"] or "unknown"
                        })
                    system_text = ""

            if chunk_mic is not None:
                user_text = self.whisper(chunk_mic["audio"])
                if user_text:
                    self.output_queue_user.put({
                        "text": user_text,
                        "direction": "center"
                        })
                    user_text = ""

            time.sleep(0.01)

    def whisper(self, chunk):
        print("whisper working")
        segments, _ = self.stt_model.transcribe(
            chunk,
            beam_size=4,
            vad_filter=False,
            condition_on_previous_text=False
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            return None

        return text

    def get_system(self):
        return self.output_queue_system.get()

    def get_mic(self):
        return self.output_queue_user.get()

    def stop(self):
        self.done = True
        slef.thread.join()

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
