import torch
import torchaudio
import pyaudio
import numpy as np
from vad_iterator import VADIterator
from df.enhance import enhance, init_df
import logging

logger = logging.getLogger(__name__)

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound

class VADHandler:
    def __init__(self, audio_enhancement=False):
        self.model, _ = torch.hub.load(
            repo_or_dir='/home/gyc/.cache/torch/hub/snakers4_silero-vad_master', # 'snakers4/silero-vad', 
            model='silero_vad', 
            trust_repo=True, 
            force_reload=False, 
            source="local"
        )
        self.sample_rate = 16000
        self.iterator = VADIterator(
            self.model,
            threshold=0.3,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=1000,
            speech_pad_ms=30
        )
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            from df.enhance import init_df
            self.enhanced_model, self.df_state, _ = init_df()

    def process_audio_chunk(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            return array
        return None

    def enhance_audio(self, array):
        if self.audio_enhancement:
            if self.sample_rate != self.df_state.sr():
                audio_float32 = torchaudio.functional.resample(
                    torch.from_numpy(array),
                    orig_freq=self.sample_rate,
                    new_freq=self.df_state.sr(),
                )
                enhanced = enhance(
                    self.enhanced_model,
                    self.df_state,
                    audio_float32.unsqueeze(0),
                )
                enhanced = torchaudio.functional.resample(
                    enhanced,
                    orig_freq=self.df_state.sr(),
                    new_freq=self.sample_rate,
                )
            else:
                enhanced = enhance(
                    self.enhanced_model, self.df_state, audio_float32
                )
            array = enhanced.numpy().squeeze()
            return array

    def setup_audio_stream(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        SAMPLE_RATE = self.sample_rate
        CHUNK = 512

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        return stream, audio

    async def record_and_process(self, listen_control, callback):
        stream, audio = self.setup_audio_stream()
        # print("listen Start")
        while True:
            audio_chunk = stream.read(512)
            processed_audio = self.process_audio_chunk(audio_chunk)
            if processed_audio is not None:
                # print("listen Stop")
                stream.stop_stream()
                # print("listen_control.clear()")
                listen_control.clear()
                enhanced_audio = self.enhance_audio(processed_audio)
                await callback(enhanced_audio)
                while not listen_control.is_set():
                    pass
                # print("listen Start")
                stream.start_stream()
                
        stream.stop_stream()
        stream.close()
        audio.terminate()
