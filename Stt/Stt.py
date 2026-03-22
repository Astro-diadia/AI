# from faiss import IndexFlatIP
from faster_whisper import WhisperModel
# from speechbrain.pretrained import EncoderClassifier
from Stt.AudioCapture import AudioCapture
from Stt.Buffer import Buffer

from time import time, sleep
from queue import Queue, put, get
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

        self.output_queue = Queue()
        self.last_speech = time()
        self.silence_time = 0.6

        self.done = False
        self.worker = threading.Thread(
            target=self.stt_worker,
            daemon=True
        )
        self.worker.start()

    def stt_worker(self):
        while not self.done:
            chunk_mic, direction = self.mic_buffer.process_chunk(self.audio.get_mic_audio())
            chunk_system, direction = self.system_buffer.process_chunk(self.audio.get_system_audio())

            if chunk_mic is not None:
                process_mic(chunk_mic["audio"])

            if chunk_system is not None:
                process_system(chunk_system["audio"])

            if (self.accumulated_text.strip()
                and (time() - self.last_speech) > self.silence_time
                ):


                self.output_queue.put({
                    "system_text": system_text,
                    "user_text": user_text,
                    "direction": chunk_system["direction"]
                    })

                self.accumulated_text = ""

            sleep(0.01)

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

    def process_system(self, chunk):
        text = self.wisper(chunk)

        if text:
            return {
                "text": text,
                "direction": direction,
                "source": "system"
            }

        return None

    def process_mic(self, chunk):
        text = self.wisper(chunk)

        if text:
            return {
                "text": text,
                "direction": "center",
                "source": "mic"
            }

        return None

    def get(self):


    def stop(self):
        self.done = True
        slef.thread.join()
