from collections import defaultdict, deque
from typing import List, Dict
from time import time

# py "D:\MemControl\ShortMemory.py"

class ShortMemory:
    def __init__(self, max_speaker_messages=10):
        self.max_speaker_messages = max_speaker_messages
        self.buffers = defaultdict(lambda: deque())
        self.timestamps = defaultdict(lambda: deque())

    def add(
        self,
        content: str,
        speaker_id: str,
        role: str,
        mid_memory = None
    ):
        buf = self.buffers[speaker_id]
        ts_buf = self.timestamps[speaker_id]

        buf.append({"role": role, "content": content, "speaker": speaker_id})
        ts_buf.append(time())

        if len(buf) > self.max_speaker_messages:
            overflow_item = buf.popleft()
            overflow_ts = ts_buf.popleft()

            if mid_memory is not None:
                mid_memory.add(
                    role=overflow_item["role"],
                    content=overflow_item["content"],
                    speaker_id=overflow_item["speaker"],
                )

    def get(self) -> List[Dict]:
        all_items = []
        for speaker in self.buffers:
            for i, msg in enumerate(self.buffers[speaker]):
                all_items.append({
                    **msg,
                    "timestamp": self.timestamps[speaker][i]
                })

        all_items.sort(key=lambda x: x["timestamp"])
        return all_items

    def clear(self):
        self.buffers.clear()
        self.timestamps.clear()
