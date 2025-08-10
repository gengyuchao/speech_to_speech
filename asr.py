# asr.py
import whisper
import wave

class WhisperASR:
    def __init__(self, model_name="turbo"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_file, language="zh", prompt=""):
        result = self.model.transcribe(audio_file, language=language, prompt=prompt)
        return result["text"]