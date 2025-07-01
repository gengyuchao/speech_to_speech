# tts_queue.py
from queue import Queue
from threading import Thread
import sounddevice as sd
import soundfile as sf
import uuid
import os

from indextts.infer import IndexTTS
from text_cleaner import TextCleaner

tts = IndexTTS(model_dir="../index-tts/checkpoints",cfg_path="../index-tts/checkpoints/config.yaml")
voice_ref = "/home/gyc/Project/index-tts/voice/钟离.mp3"
voice_ref = "/home/gyc/Project/index-tts/voice/温迪_white.mp3"

cleaner = TextCleaner()


# 创建线程安全队列
text_queue = Queue()
audio_queue = Queue()

 # 初始化计数器
tts_worker_counter = 1

tts_kwargs = {
    "do_sample": True,
    "top_p": float(0.8),
    "top_k": int(30),
    "temperature": float(1),
    "length_penalty": float(0),
    "num_beams": 3,
    "repetition_penalty": float(10),
    "max_mel_tokens": int(600),
    # "typical_sampling": bool(typical_sampling),
    # "typical_mass": float(typical_mass),
}

from pydub import AudioSegment
from pydub.silence import detect_silence
import argparse

def calculate_silence_ratio(file_path, silence_threshold=-50, min_silence_len=1):
    """
    计算音频文件中静音时间占总时长的比例。

    :param file_path: 音频文件路径
    :param silence_threshold: 静音的 dBFS 阈值（默认 -50）
    :param min_silence_len: 最小静音段长度（毫秒，默认 1）
    :return: 静音时间占比（0 到 1 之间）
    """
    # 读取音频文件
    audio = AudioSegment.from_file(file_path)

    # 获取音频总时长（毫秒）
    total_duration = len(audio)

    if total_duration == 0:
        return 0.0

    # 检测所有静音段
    silence_periods = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=1  # 检测步长（毫秒）
    )

    # 计算所有静音段的总时长
    total_silence = sum(end - start for start, end in silence_periods)

    # 计算静音时间占比
    return total_silence / total_duration

def tts_worker():
    # 使用 nonlocal 或 global 来维护计数器
    global tts_worker_counter 
    while True:
        text = text_queue.get()
        if text is None:
            break
        # 格式化序号，保留4位数字前缀
        filename = f"{tts_worker_counter:04d}_{uuid.uuid4()}.wav"
        output_path = f"cache/{filename}"
        print(f"[生成语音]:{text}")
        tts.infer(voice_ref, text, output_path, max_text_tokens_per_sentence=120, **tts_kwargs)
        print(f"[语音生成完毕]:{text}")
        
        while calculate_silence_ratio(output_path) > 0.5 :
            print(f"发现异常语音，静音比例{calculate_silence_ratio(output_path)}，重新生成")
            print(f"[生成语音]:{text}")
            tts.infer(voice_ref, text, output_path, max_text_tokens_per_sentence=120, **tts_kwargs)
            print(f"[语音生成完毕]:{text}")

        audio_queue.put(output_path)
        tts_worker_counter += 1  # 计数器递增
        text_queue.task_done()

def play_worker():
    while True:
        path = audio_queue.get()
        if path is None:
            break
        data, samplerate = sf.read(path)
        print(f"[语音播放]:{path}")
        sd.play(data, samplerate)
        sd.wait()
        print(f"[语音完毕]:{path}")
        # os.remove(path)
    audio_queue.task_done()

def start_tts_threads():
    tts_thread = Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    return tts_thread # , play_thread

def submit_text(text):
    cleaned_text = cleaner.clean(text)
    if cleaned_text:
        text_queue.put(cleaned_text)
    else:
        print("Clean Text cause not text to submit.")

def stop_tts_threads():
    text_queue.put(None)

def start_play_threads():
    play_thread = Thread(target=play_worker, daemon=True)
    play_thread.start()
    return play_thread

def stop_play_threads():
    audio_queue.put(None)
