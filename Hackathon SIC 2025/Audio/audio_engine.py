import time
import os
from gtts import gTTS
from playsound import playsound


class AudioEngine:
    def __init__(self, cooldown=3):
        self.cooldown = cooldown
        self.last_spoken = 0
        self.audio_file = "temp_audio.mp3"

    def speak(self, text):
        now = time.time()

        if not text:
            return

        if now - self.last_spoken < self.cooldown:
            return

        try:
            tts = gTTS(text=text, lang="es", slow=False)
            tts.save(self.audio_file)
            playsound(self.audio_file)
        except Exception as e:
            print("Error de audio:", e)

        self.last_spoken = now