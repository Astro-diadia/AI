# from faiss import IndexFlatIP
from faster_whisper import WhisperModel
# from speechbrain.pretrained import EncoderClassifier
from Stt.AudioCapture import AudioCapture
from Stt.Buffer import Buffer
from time import time
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

        self.system_buffer = Buffer(0.05)
        self.mic_buffer = Buffer(0.015)


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

    def wisper(self, source, chunk):
        # print(self.stt_model.model.device)
        # print(self.stt_model.model.compute_type)
        segments, _ = self.stt_model.transcribe(
            chunk,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            # language="en",
            # language="ru"
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            return None

        return text

    def process_system(self):
        chunk = self.system_buffer.process_block(self.audio.get_system_audio())

        if chunk is None:
            return None

        return self.wisper("system", chunk)

    def process_mic(self):
        chunk = self.mic_buffer.process_block(self.audio.get_mic_audio())

        if chunk is None:
            return None

        return self.wisper("mic", chunk)

