import torch
import threading
import queue
import sounddevice as sd

# py "D:\AI\Tts\Tts2.py"

class Tts:
    def __init__(self, language="ru", model_id="v5_ru", speaker="baya", sample_rate=24000, device="cpu"):
        self.language = language
        self.model_id = model_id
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self.language,
            speaker=self.model_id,
            # speaker='random',
            # voice_path="D:\\",
            put_accent=True,
            put_yo=True,
            put_stress_homo=True,
            put_format='stream',
            # put_yo_homo=True,
        )

        self.model.to(self.device)

        self.queue = queue.Queue()

        self.output_queue = queue.Queue()

        self.worker = threading.Thread(
            target=self.tts_worker,
            daemon=True
        )
        self.worker.start()

    def tts_worker(self):
        while True:
            text = self.queue.get()

            if text is None:
                break

            audio = self.model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate
            )

            self.output_queue.put(audio)

            self.queue.task_done()

    def speak(self, text):
        self.queue.put(text)

        audio = self.output_queue.get()

        return audio

# TTS =TTS()
# text = """
# <speak>
# Привет. <break time="300ms"/>
# <prosody rate="fast">Это тест новой версии силеро.</prosody>
# </speak>
# """

# audio = TTS.speak(text)

# sd.play(audio, 24000)
# sd.wait()

# Тег	Аргумент	Значения (Min / Mid / Max)	Описание
# <speak>	—	—	Корневой тег, обязателен для SSML.
# <break />	time	10ms / 500ms / 5000ms	Пауза. Рекомендуется не превышать 5 сек.
# strength	x-weak, weak, medium, strong, x-strong	Пресеты длительности пауз.
# <prosody>	rate	x-slow (50%) / medium (100%) / x-fast (200%)	Скорость речи. Можно в % (напр. 85%).
# pitch	x-low / medium / x-high	Высота тона. Влияет на «эмоциональную» окраску.
# volume	silent, x-soft, medium, loud, x-loud	Относительная громкость фрагмента.
# <emphasis>	level	reduced, moderate, strong	Степень логического ударения на слове.