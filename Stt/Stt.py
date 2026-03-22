# from faiss import IndexFlatIP
from faster_whisper import WhisperModel
# from speechbrain.pretrained import EncoderClassifier
from Stt.AudioCapture import AudioCapture
from Stt.Buffer import Buffer

from time import time
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

        self.prev_ratio = 0.0
        self.buffer_direction = []
        self.direction = "center"

        self.system_buffer = Buffer(0.05)
        self.mic_buffer = Buffer(0.015)

        self.output_queue = Queue()
        self.last_speech_time = time.time()
        self.silence_threshold = 0.6

        self.done = False
        self.worker = threading.Thread(
            target=self.stt_worker,
            daemon=True
        )
        self.worker.start()

    def stt_worker(self):
        while not self.done:
            chunk_system = self.system_buffer.process_chunk(self.audio.get_system_audio())
            chunk_mic = self.mic_buffer.process_chunk(self.audio.get_mic_audio())

            if chunk_mic is not None:
                process_mic(chunk_mic)

            if chunk_system is not None:
                process_system(chunk_system)

            now = time.time()
            if (
                self.accumulated_text.strip()
                and (now - self.last_speech_time) > self.silence_threshold
            ):
                final_text = self.accumulated_text.strip()
                self.output_queue.put(final_text)

                self.accumulated_text = ""

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

    def classify_direction(self, chunk, threshold=0.1):
        self.buffer_direction.append(chunk)

        if len(self.buffer_direction) >= 8:
            big_chunk = np.concatenate(self.buffer_direction)

            left = np.abs(big_chunk[:, 0]).mean()
            right = np.abs(big_chunk[:, 1]).mean()

            ratio = (left - right) / (left + right + 1e-6)

            smoothed_ratio = 0.8 * self.prev_ratio + 0.2 * ratio

            self.prev_ratio = smoothed_ratio

            if smoothed_ratio > threshold:
                return "left"
            elif smoothed_ratio < 1 / threshold:
                return "right"
            else:
                return "center"
        return None

    def process_system(self, chunk):
        text = self.wisper(chunk)

        direction_local = self.classify_direction(chunk)

        self.direction = direction_local
        self.buffer_direction.clear()

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

    def stop(self):
        self.done = True
        slef.thread.join()
