#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# audio_manager.py

"""
音频管理模块
"""
import pyaudio
import numpy as np
import torch
import threading
import queue
import wave
from collections import deque
from vad_controller import VadController
from asr import WhisperASR, TransformersASR, FasterWhisperASR
from tts_playback import stop_playback_flag, audio_queue, text_queue
from logger_config import system_logger
from config_manager import config_manager

# 从配置文件导入参数
# 使用统一配置管理器

FORMAT = pyaudio.paInt16
CHANNELS = config_manager.get('audio.channels')
RATE = config_manager.get('audio.rate')
CHUNK = config_manager.get('audio.chunk')
SILENCE_FRAME_THRESHOLD = config_manager.get('audio.silence_frame_threshold')

class AudioManager:
    def __init__(self, result_queue: queue.Queue):
        self.result_queue = result_queue
        self.running = False
        self.vad_controller = VadController()
        # self.asr = WhisperASR()
        # self.asr = TransformersASR()
        self.asr = FasterWhisperASR()

        

    def start_listening(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop)
        self.thread.start()
        
    def stop_listening(self):
        self.running = False
        self.thread.join()
        
    def set_vad_sensitivity(self, sensitivity: float):
        """设置 VAD 敏感度"""
        self.vad_controller.set_sensitivity(sensitivity)
            
    def _listen_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        system_logger.info("开始监听...")
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            source='local'
        )
        
        get_speech_timestamps, _,_ , _,_ = utils
        audio_buffer = deque(maxlen=int(RATE / CHUNK * 2))
        speech_buffer = []
        head_buffer = []
        recording = False
        silence_frames = 0
        
        while self.running:
            try:
                data = stream.read(CHUNK)
            except Exception as e:
                system_logger.error("音频读取错误：{}".format(e))
                break
                
            # 将音频数据转换为 numpy 数组
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 使用原始音频进行 VAD 检测
            audio_tensor = torch.tensor(audio_np)
            threshold = self.vad_controller.get_threshold()
            
            with torch.no_grad():
                prob = model(audio_tensor, RATE).item()
                
            if prob > threshold:
                if not recording:
                    system_logger.info("检测到说话开始... (概率: {:.3f}, 阈值: {:.3f})".format(prob, threshold))
                    speech_buffer = []
                    head_buffer = list(audio_buffer)
                    recording = True
                    
                    # 真正的用户语音，执行打断操作
                    self._interrupt_tts()
                        
                speech_buffer.append(data)
                silence_frames = 0
            else:
                if recording:
                    silence_frames += 1
                    speech_buffer.append(data)
                    
                    if silence_frames >= SILENCE_FRAME_THRESHOLD:
                        system_logger.info("检测到说话结束，开始识别...")
                        
                        final_audio = list(head_buffer) + speech_buffer


                        # 保存为 WAV 文件
                        filename = "temp.wav"
                        with wave.open(filename, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(final_audio))
                            
                        # 调用 Whisper 识别
                        try:
                            text = self.asr.transcribe(
                                filename,
                                language="zh",
                                prompt=config_manager.get('asr_prompt')
                            )
                            system_logger.info("识别结果：{}".format(text))
                            self.result_queue.put(text)
                        except Exception as e:
                            system_logger.error("识别失败：{}".format(e))
                                
                        # 重置状态
                        recording = False
                        speech_buffer = []
                        silence_frames = 0
                        audio_buffer.clear()
                else:
                    audio_buffer.append(data)
                    
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    def _interrupt_tts(self):
        clean_flag = False
        
        while not text_queue.empty():
            clean_flag = True
            try:
                text_queue.get_nowait()
            except queue.Empty:
                break
                
        while not audio_queue.empty():
            clean_flag = True
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
                
        if clean_flag:
            system_logger.info("停止播放进程")
            stop_playback_flag.set()