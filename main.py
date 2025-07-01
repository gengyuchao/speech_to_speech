# -*- coding: utf-8 -*-
import ollama_stream4
import tts_playback
import queue
import sys
import threading
import time

# 确保中文输入输出正常
import readline

# 导入语音识别模块
from speech_recognition_task import SpeechRecognitionTask


def voice_input_loop(result_queue):
    """用于启动语音识别任务的函数"""
    recognizer = SpeechRecognitionTask(result_queue)
    recognizer.start()
    print("语音识别已启动，你可以开始说话...")


def process_input(text_queue, result_queue):
    """处理输入，无论来自键盘还是语音"""
    while True:
        source, content = None, None

        # 检查队列中是否有数据
        try:
            item = text_queue.get(timeout=0.1)  # 使用 timeout 防止阻塞
            source, content = item
        except queue.Empty:
            pass

        try:
            content = result_queue.get(timeout=0.1)
            source = 'voice'
        except queue.Empty:
            pass

        if source is None:
            time.sleep(0.01)  # 避免 CPU 占用过高
            continue

        if source == 'exit':
            break

        if content:
            print(f"\n收到{'语音' if source == 'voice' else '文字'}输入：{content}")
            try:
                ollama_stream4.chat_handle(content, "玉超", tts_playback.submit_text)
                # ollama_stream4.chat_handle(content, "玉超", None)
            except Exception as e:
                print("LLM 处理失败：", e)


def main():
    # 初始化队列
    text_queue = queue.Queue()
    result_queue = queue.Queue()

    # 启动 TTS 和播放线程
    tts_playback.start_tts_threads()
    tts_playback.start_play_threads()

    # 启动语音识别线程（可选）
    voice_thread = threading.Thread(target=voice_input_loop, args=(result_queue,), daemon=True)
    voice_thread.start()

    # 启动处理输入线程
    process_thread = threading.Thread(target=process_input, args=(text_queue, result_queue), daemon=True)
    process_thread.start()

    # 主线程处理键盘输入，保证 input() 可正常显示
    try:
        while True:
            user_prompt = input("请输入你的问题（按 q 退出）：")
            if user_prompt.strip().lower() in ['q', 'quit', 'exit']:
                print("正在退出...")
                text_queue.put(('exit', None))
                break

            if not user_prompt.strip():
                print("输入不能为空，请重新输入。")
                continue

            if all(not c.isalnum() for c in user_prompt):
                print("输入内容无效，请输入有意义的问题。")
                continue

            text_queue.put(('text', user_prompt))

    except KeyboardInterrupt:
        print("\n用户中断，退出中...")
        text_queue.put(('exit', None))
    finally:
        tts_playback.stop_tts_threads()
        tts_playback.stop_play_threads()
        sys.exit(0)


if __name__ == "__main__":
    main()