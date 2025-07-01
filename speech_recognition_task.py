import torch
import torchaudio
import pyaudio
import wave
import whisper
import numpy as np
from collections import deque
import threading
import queue

# ===== 配置参数 =====
FORMAT = pyaudio.paInt16      # 音频格式（16位PCM）
CHANNELS = 1                 # 单声道
RATE = 16000                 # 采样率（Silero VAD 支持 16kHz 或 8kHz）
CHUNK = 512                  # 每次读取的音频块大小（建议 512-2048）
SILENCE_DURATION_THRESHOLD = 1.5  # 静默超过1.5秒才判定结束
SILENCE_FRAME_THRESHOLD = 30
MIN_SPEECH_DURATION = 0.3    # 最小语音持续时间（过滤短暂噪声）

# ===== 加载 Silero VAD 模型 =====
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, source='local')
(get_speech_timestamps, _, _, _, _) = utils

# ===== 加载 Whisper 模型 =====
whisper_model = whisper.load_model("turbo")

# ===== 音频预处理函数 =====
def process_audio(data):
    # 将音频数据转换为 PyTorch 张量并归一化到 [-1, 1]
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.tensor(audio)

class SpeechRecognitionTask(threading.Thread):
    def __init__(self, result_queue: queue.Queue):
        super().__init__()
        self.result_queue = result_queue
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("开始监听...")

        audio_buffer = deque(maxlen=int(RATE / CHUNK * 2))  # 缓存最近2秒音频
        speech_buffer = []
        head_buffer = []
        recording = False
        silence_frames = 0

        while self.running:
            try:
                data = stream.read(CHUNK)
            except Exception as e:
                print(f"音频读取错误：{e}")
                break

            audio_tensor = process_audio(data)

            # 使用 Silero VAD 检测语音活动
            with torch.no_grad():
                speech_probs = model(audio_tensor, RATE).item()

            if speech_probs > 0.3:
                if not recording:
                    print("检测到说话开始...")
                    speech_buffer = []
                    head_buffer = list(audio_buffer)  # 浅拷贝，保存检测前的缓存
                    recording = True
                speech_buffer.append(data)
                silence_frames = 0
            else:
                if recording:
                    silence_frames += 1
                    speech_buffer.append(data)
                    if silence_frames >= SILENCE_FRAME_THRESHOLD:
                        # 静默超过阈值，判定语音结束
                        print("检测到说话结束，开始识别...")
                        # 合并缓存中的前导静默部分（提升识别效果）
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
                            result = whisper_model.transcribe(filename, 
                                                            language="zh", 
                                                            prompt="这是钟离和玉超在进行的人工智能方面的技术讨论，其中包括 whisper 模型的内容")
                            text = result["text"]
                            print("识别结果：", text)
                            # 把识别结果放入队列
                            self.result_queue.put(text)
                        except Exception as e:
                            print("识别失败：", e)

                        # 重置状态
                        recording = False
                        speech_buffer = []
                        silence_frames = 0
                        audio_buffer.clear()
                else:
                    audio_buffer.append(data)

        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()