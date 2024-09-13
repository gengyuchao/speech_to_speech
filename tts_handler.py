import queue
import threading
import edge_tts  # 假设你已经安装了 edge_tts 库
import pyaudio    # 用于播放音频
import wave
from pydub import AudioSegment
from pydub.playback import play

# 创建两个 Queue
sentence_queue = queue.Queue()
audio_queue = queue.Queue()

# 创建一个 Event 用于控制任务是否结束
stop_event = threading.Event()

VOICE = "zh-CN-XiaoxiaoNeural"

# 任务1：从句子队列中获取句子，生成音频并放入音频队列
def generate_audio():
    mp3_count = 0
    while not stop_event.is_set():
        try:
            sentence = sentence_queue.get(timeout=1)  # 从句子队列中获取句子，超时时间为1秒
            if sentence is None:  # 如果接收到 None，表示任务结束
                break
            # 使用 edge_tts 生成音频
            # print(f"sentence[{sentence}]")
            communicate = edge_tts.Communicate(sentence, VOICE)
            file_name = "voice_history/v" + str(mp3_count) + ".mp3"
            communicate.save_sync(file_name)
            audio_queue.put(file_name)  # 将音频放入音频队列
            mp3_count = mp3_count + 1
            # print(f"mp3_count:{mp3_count}")
        except queue.Empty:
            continue  # 如果没有任务，继续等待


# 任务2：从音频队列中获取音频并播放
def play_audio(listen_control,generating):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    output=True)

    while not stop_event.is_set():
        try:
            audio_path = audio_queue.get(timeout=2)  # 从音频队列中获取音频，超时时间为1秒
            if audio_path is None:  # 如果接收到 None，表示任务结束
                break
            
            # 读取临时文件中的音频数据

            # 将 MP3 文件转换为 WAV 文件
            sound = AudioSegment.from_mp3(audio_path)

            # 播放音频
            play(sound)
            
            if audio_queue.empty() and sentence_queue.empty() and (not generating.is_set()):
                listen_control.set()
                print("listening...")
        
        except queue.Empty:
            continue  # 如果没有任务，继续等待
    
    # 关闭流和 PyAudio 实例
    stream.stop_stream()
    stream.close()
    p.terminate()
    
# # 启动两个线程
# threading.Thread(target=generate_audio).start()
# threading.Thread(target=play_audio).start()

import pyttsx3

# 任务3：从句子队列中获取句子，本地生成音频并播放
def generate_local_audio(listen_control,generating):
    engine = pyttsx3.init()  # 初始化TTS引擎

    # 设置使用的语音包
    engine.setProperty('voice', 'zh') #开启支持中文
    engine.setProperty('volume', 0.7)
    engine.setProperty('rate', 200)


    while not stop_event.is_set():
        try:
            sentence = sentence_queue.get(timeout=1)  # 从句子队列中获取句子，超时时间为1秒
            if sentence is None:  # 如果接收到 None，表示任务结束
                continue

            if sentence.strip() == "":
                print(f"sentence empty")
                continue


            # print(f"sentence:[{sentence}]")
            engine.say(sentence)  # 将文本转换为语音
            engine.runAndWait()
            if sentence_queue.empty() and not generating.is_set() :
                listen_control.set()
                print("listening...")
            # print(f"mp3_count:{mp3_count}")
        except queue.Empty:
            continue  # 如果没有任务，继续等待
