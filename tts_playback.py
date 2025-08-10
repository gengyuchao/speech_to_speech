# tts_queue.py
import sys
from queue import Queue
from threading import Thread
import sounddevice as sd
import soundfile as sf
import uuid
import os
import time
import yaml
import threading

from indextts.infer import IndexTTS
from text_cleaner import TextCleaner


# 加载配置文件
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 提取配置项
tts_model_dir = config["tts_model_dir"]
tts_cfg_path = config["tts_cfg_path"]
cache_dir = config["cache_dir"]
tts_kwargs = config["tts_kwargs"]
SPEAKER_VOICES = config["speaker_voices"]
vad_sensitivity = config["vad"]["sensitivity"]
silence_threshold = config["silence_detection"]["silence_threshold"]
min_silence_len = config["silence_detection"]["min_silence_len"]
worker_counter_start = config["worker_counter_start"]

cleaner = TextCleaner()

# 创建线程安全队列
text_queue = Queue()
audio_queue = Queue()

stop_playback_flag = threading.Event()

# 全局 AudioManager 引用
g_audio_manager = None

def set_audio_manager(am):
    """设置 AudioManager 实例"""
    global g_audio_manager
    g_audio_manager = am

# 初始化计数器
tts_worker_counter = 1


# 初始化TTS引擎
tts = IndexTTS(model_dir=tts_model_dir, cfg_path=tts_cfg_path)

from pydub import AudioSegment
from pydub.silence import detect_silence
import argparse
import numpy as np
import librosa

def play_audio_with_similarity_detection(audio_data, sample_rate=22050):
    """播放音频并将其发送给相似度检测系统"""
    global g_audio_manager
    
    try:
        # 如果采样率不同，需要重采样到16kHz
        if sample_rate != 16000:
            # 确保输入是float32类型
            if audio_data.dtype != np.float32:
                audio_float = audio_data.astype(np.float32)
            else:
                audio_float = audio_data
                
            audio_data_16k = librosa.resample(audio_float, 
                                            orig_sr=sample_rate, target_sr=16000)
        else:
            if audio_data.dtype != np.float32:
                audio_data_16k = audio_data.astype(np.float32)
            else:
                audio_data_16k = audio_data
        
        # 发送音频数据给相似度检测系统
        if g_audio_manager:
            try:
                g_audio_manager.add_playback_audio(audio_data_16k)
            except Exception as e:
                print(f"发送音频到相似度检测失败: {e}")
    except Exception as e:
        print(f"音频处理错误: {e}")


def set_vad_playing(playing):
    """设置VAD播放状态"""
    global g_audio_manager
    if g_audio_manager:
        try:
            # 使用异步方式调用
            g_audio_manager.vad_controller.set_playing(playing)
        except Exception as e:
            print(f"设置VAD播放状态失败: {e}")

def set_vad_sensitivity(sensitivity):
    """设置VAD敏感度"""
    global g_audio_manager
    if g_audio_manager:
        try:
            # 使用异步方式调用
            g_audio_manager.vad_controller.set_sensitivity(sensitivity)
        except Exception as e:
            print(f"设置VAD敏感度失败: {e}")

def calculate_silence_ratio(file_path, silence_threshold=-50, min_silence_len=1):
    """
    计算音频文件中静音时间占总时长的比例。
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
        seek_step=1
    )

    # 计算所有静音段的总时长
    total_silence = sum(end - start for start, end in silence_periods)

    # 计算静音时间占比
    return total_silence / total_duration

def tts_worker():
    global tts_worker_counter 
    while True:
        # 修改：从队列获取包含文本和说话人信息的元组
        item = text_queue.get()
        if item is None:
            break
            
        text, response_speaker = item
        
        # 根据说话人选择对应的音色文件
        voice_ref = SPEAKER_VOICES.get(response_speaker, SPEAKER_VOICES["unknown"])
        
        # 格式化序号，保留4位数字前缀
        filename = f"{tts_worker_counter:04d}_{uuid.uuid4()}.wav"
        output_path = f"cache/{filename}"
        print(f"[生成语音] 说话人: {response_speaker}, 文本: {text}")
        tts.infer(voice_ref, text, output_path, max_text_tokens_per_sentence=120, **tts_kwargs)
        # print(f"[语音生成完毕] 说话人: {response_speaker}, 文本: {text}")
        
        while calculate_silence_ratio(output_path) > 0.5 :
            print(f"发现异常语音，静音比例{calculate_silence_ratio(output_path)}，重新生成")
            # print(f"[生成语音] 说话人: {response_speaker}, 文本: {text}")
            tts.infer(voice_ref, text, output_path, max_text_tokens_per_sentence=120, **tts_kwargs)
            # print(f"[语音生成完毕] 说话人: {response_speaker}, 文本: {text}")

        audio_queue.put(output_path)
        tts_worker_counter += 1
        text_queue.task_done()

def play_worker():
    print("播放线程循环启动")
    while True:
        path = audio_queue.get()
        if path is None:
            break

        # 检查是否需要中断播放
        if stop_playback_flag.is_set():
            print("[播放中断] 用户开始说话，跳过当前语音播放")
            stop_playback_flag.clear()
            continue

        # 设置播放状态，降低VAD敏感度
        # print("设置播放状态，降低VAD敏感度")
        set_vad_playing(True)

        try:
            # print(f"[语音播放]:{path}")
            data, samplerate = sf.read(path)
            
            # 将播放的音频发送给相似度检测系统
            play_audio_with_similarity_detection(data, samplerate)

            # 临时重定向 stderr
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')

            # 开始播放
            sd.play(data, samplerate)
 

            # 等待播放完成，但可以被中断
            while sd.get_stream().active:
                if stop_playback_flag.is_set():
                    sd.stop()
                    print("[播放中断] 用户开始说话，立即停止播放")
                    stop_playback_flag.clear()
                    break
                time.sleep(0.01)

           
            # 恢复 stderr
            sys.stderr.close()
            sys.stderr = old_stderr

            if not stop_playback_flag.is_set():
                # print(f"[语音完毕]:{path}")
                pass
                
        except Exception as e:
            print(f"[播放错误]: {e}")
            try:
                sd.stop()
            except:
                pass
        finally:
            # 恢复正常VAD敏感度
            set_vad_playing(False)

    print("播放线程循环结束")

def start_tts_threads():
    print("启动TTS线程")
    tts_thread = Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    return tts_thread

def submit_text(text, response_speaker="unknown"):
    """
    提交文本到TTS队列
    
    Args:
        text (str): 要转换的文本
        response_speaker (str): 回答的说话人，默认为 "unknown"
    """
    cleaned_text = cleaner.clean(text)
    if cleaned_text:
        # 修改：将文本和说话人信息作为一个元组放入队列
        text_queue.put((cleaned_text, response_speaker))
    else:
        print("Clean Text cause not text to submit.")

def stop_tts_threads():
    text_queue.put(None)

def start_play_threads():
    print("启动播放线程")
    play_thread = Thread(target=play_worker, daemon=True)
    play_thread.start()
    return play_thread

def stop_play_threads():
    audio_queue.put(None)