# -*- coding: utf-8 -*-
import asyncio
import ollama_stream4
import tts_playback
import queue
import sys
import threading

# 确保中文输入输出正常
import readline

# 导入语音识别模块
from speech_recognition_task import SpeechRecognitionTask


def voice_input_loop(result_queue):
    """用于启动语音识别任务的函数"""
    recognizer = SpeechRecognitionTask(result_queue)
    recognizer.start()
    print("语音识别已启动，你可以开始说话...")


async def process_input_async(text_queue, result_queue):
    """异步处理输入，无论来自键盘还是语音"""
    while True:
		# 尝试从两个队列中读取输入
        source, content = None, None

        try:
            item = text_queue.get_nowait()
            source, content = item
        except queue.Empty:
            pass

        try:
            content = result_queue.get_nowait()
            source = 'voice'
        except queue.Empty:
            pass

        if source is None:
            await asyncio.sleep(0.1)  # 避免 CPU 占用过高
            continue

        if source == 'exit':
            break

        if content:
            print(f"\n收到{'语音' if source == 'voice' else '文字'}输入：{content}")
            try:
                await ollama_stream4.chat_handle(content, "玉超", tts_playback.submit_text)
                # await ollama_stream4.chat_handle(content, "玉超", None)
            except Exception as e:
                print("LLM 处理失败：", e)


def async_loop_worker(text_queue, result_queue):
    """异步事件循环放到子线程中运行"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_input_async(text_queue, result_queue))
    except Exception as e:
        print("Async loop error:", e)
    finally:
        loop.close()


def main():
    # 初始化队列
    text_queue = queue.Queue()
    result_queue = queue.Queue()

    # # 启动 TTS 和播放线程
    tts_playback.start_tts_threads()
    tts_playback.start_play_threads()

    # 启动异步处理线程（放到后台线程）
    async_thread = threading.Thread(target=async_loop_worker, args=(text_queue, result_queue), daemon=True)
    async_thread.start()

    # 启动语音识别线程（可选）
    voice_thread = threading.Thread(target=voice_input_loop, args=(result_queue,), daemon=True)
    voice_thread.start()

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
