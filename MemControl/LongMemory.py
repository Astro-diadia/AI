from os import path
from json import load, dump
from time import time
from faiss import IndexFlatL2, read_index, write_index
import numpy as np
from sentence_transformers import SentenceTransformer

# py "D:\MemControl\LongMemory.py"

class LongMemory:
    def __init__(
        self,
        dim=384,
        model_name="all-MiniLM-L6-v2",
        index_path=r"D:\FAISS\faiss.index",
        meta_path=r"D:\FAISS\meta.json" 
    ):
        self.dim = dim
        self.model = SentenceTransformer(model_name)

        self.index_path = index_path
        self.meta_path = meta_path

        if path.exists(self.index_path):
            self.index = read_index(self.index_path)
        else:
            self.index = IndexFlatL2(dim)

        if path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = load(f)
        else:
            self.meta = []

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True)
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        return vec.astype("float32")

    def retrieve_similar(self, query: str, how_much_items_to_retrive=5):
        if self.index.ntotal == 0:
            return []

        qvec = self.embed(query)
        distances, indices = self.index.search(qvec, how_much_items_to_retrive)
        distances = distances[0].tolist()

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.meta["items"]):
                if distances[i] < 0.7:
                    results.append(self.meta["items"][idx])

        return results

    def store(self, value: str):
        vec = self.embed(value)

        self.index.add(vec)

        self.meta["items"].append({
            "id": len(self.meta["items"]),
            "value": value,
            "timestamp": time(),
        })

        write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            dump(self.meta, f, indent=2)
