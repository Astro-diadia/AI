from faiss import IndexFlatL2
import numpy as np 
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from time import time

# py "D:\MemControl\MidMemory.py"

class MidMemory:
    def __init__(self, dim=384, model_name="all-MiniLM-L6-v2"):
        self.dim = dim
        self.model = SentenceTransformer(model_name)

        self.index = IndexFlatL2(dim)

        self.items: List[Dict] = []

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True)
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        return vec.astype("float32")

    def add(self, role: str, content: str, speaker_id: str):
        text = f"{role}: {content}"
        vec = self.embed(text)

        self.index.add(vec)
        self.items.append({
            "role": role,
            "content": content,
            "speaker": speaker_id,
            "timestamp": time(),
            "text": text,
        })

    def retrieve_similar(self, query: str, how_much_items_to_retrive=3) -> List[Dict]:
        if len(self.items) == 0:
            return []

        qvec = self.embed(query)
        distances, indices = self.index.search(qvec, how_much_items_to_retrive)

        distances = distances[0].tolist()

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.items):
                if distances[i] < 0.7:
                    results.append(self.items[idx])

        return results

    def get(self) -> List[Dict]:
        return self.items

    def clear(self):
        self.index.reset()
        self.items.clear()
