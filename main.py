# -*- coding: utf-8 -*-
import ollama_stream
import tts_playback
import queue
import sys
import threading
import time

# 确保中文输入输出正常
import readline

# 导入新的语音管理模块
from audio_manager import AudioManager


def process_input(text_queue, voice_result_queue):
    """处理输入，无论来自键盘还是语音"""
    while True:
        source, content = None, None

        # 检查键盘输入队列
        try:
            item = text_queue.get(timeout=0.1)  # 使用 timeout 防止阻塞
            source, content = item
        except queue.Empty:
            pass

        # 检查语音识别结果队列
        try:
            content = voice_result_queue.get(timeout=0.1)
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
                ollama_stream.chat_handle(content, "玉超", tts_playback.submit_text)
                # ollama_stream.chat_handle(content, "玉超", None)
            except Exception as e:
                print("LLM 处理失败：", e)


def main():
    # 初始化队列
    text_queue = queue.Queue()          # 键盘输入队列
    voice_result_queue = queue.Queue()  # 语音识别结果队列

    # 启动 TTS 和播放线程
    tts_playback.start_tts_threads()
    tts_playback.start_play_threads()

    # 创建并启动音频管理器（包含语音识别功能）
    audio_manager = AudioManager(voice_result_queue)

    # 将 AudioManager 实例传递给 tts_queue
    tts_playback.set_audio_manager(audio_manager)

    audio_manager.start_listening()
    print("语音识别已启动，你可以开始说话...")
    
    # 启动处理输入线程
    process_thread = threading.Thread(target=process_input, args=(text_queue, voice_result_queue), daemon=True)
    process_thread.start()

    # 主线程处理键盘输入，保证 input() 可正常显示
    try:
        while True:
            user_prompt = input("请输入你的问题（按 q 退出，v+数字 调节敏感度）：")
            
            # 处理VAD敏感度调节命令
            if user_prompt.strip().lower().startswith('v'):
                try:
                    sensitivity = float(user_prompt.strip().lower()[1:])
                    if 0 <= sensitivity <= 1:
                        # 直接调用 audio_manager 的方法来设置敏感度
                        audio_manager.set_vad_sensitivity(sensitivity)
                        print(f"VAD敏感度已设置为: {sensitivity}")
                    else:
                        print("敏感度值应在 0-1 之间")
                except ValueError:
                    print("请输入有效的数字，例如: v0.3")
                continue
            
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
        # 停止音频管理器
        audio_manager.stop_listening()
        
        # 停止 TTS 和播放线程
        tts_playback.stop_tts_threads()
        tts_playback.stop_play_threads()
        sys.exit(0)


if __name__ == "__main__":
    main()