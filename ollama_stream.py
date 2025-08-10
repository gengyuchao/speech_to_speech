import os
import json
import ollama
from concurrent.futures import ThreadPoolExecutor
from sentence_segmenter import SentenceSegmenter
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# 全局工具初始化
segmenter = SentenceSegmenter()
executor = ThreadPoolExecutor(max_workers=1)

# 默认模型配置
default_model = "gemma3:27b"


class ChatHistoryManager:
    def __init__(self, model: str = "deepseek-r1", max_history: int = 20, compress_interval: int = 5):
        """
        Args:
            model: 用于生成摘要的模型名称
            max_history: 历史记录最大长度
            compress_interval: 每隔多少条进行一次压缩
        """
        self.model = model
        self.history: list[dict[str, str]] = []
        self.max_history = max_history
        self.compress_interval = compress_interval
        self.total_turns = 0  # 总对话轮数

    def add_message(self, role: str, content: str):
        """添加单条消息到历史"""
        self.history.append({"role": role, "content": content})
        self.total_turns += 1

    def summarize_earliest(self, m: int = 5) -> Optional[str]:
        """用模型总结最早 m 条历史记录"""
        if len(self.history) < m:
            return None

        to_summarize = self.history[:m]
        summary_prompt = (
            "请对以下对话历史进行简洁的总结，保留关键信息和上下文关系。\n\n"
            + "\n".join([f"{msg['role']}: {msg['content']}" for msg in to_summarize])
        )

        try:
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": summary_prompt}])
            summary = response["message"]["content"]
            # 替换掉前 m 条为总结
            self.history = [{"role": "system", "content": f"[历史摘要] {summary}"}] + self.history[m:]
            return summary
        except Exception as e:
            print(f"历史总结失败: {e}")
            return None

    def maybe_compress_history(self):
        """判断是否需要压缩历史"""
        if self.total_turns >= self.max_history and (self.total_turns % self.compress_interval == 0):
            self.summarize_earliest(m=self.compress_interval)

    def get_messages_for_model(self) -> list[dict[str, str]]:
        """获取当前历史作为模型输入"""
        return self.history.copy()

    def save_to_file(self, filepath: str = "history.json"):
        """将当前历史记录保存到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "total_turns": self.total_turns,
                "history": self.history
            }, f, ensure_ascii=False, indent=2)
        print(f"[+] 已保存历史到 {filepath}")

    def load_from_file(self, filepath: str = "history.json"):
        """从文件加载历史记录"""
        if not os.path.exists(filepath):
            print(f"[!] 文件 {filepath} 不存在，跳过加载")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.total_turns = data.get("total_turns", 0)
            self.history = data.get("history", [])
            print(f"[+] 已从 {filepath} 加载历史记录")
        except Exception as e:
            print(f"[!] 加载历史出错: {e}")


# 全局历史管理器实例
history_manager = ChatHistoryManager(model=default_model, max_history=20, compress_interval=5)
history_manager.load_from_file()  # 启动时尝试加载

# 获取当前本地时间
local_time = None
formatted_local = None


def stream_chat(prompt: str, model: str = default_model, speaker_id: str = "unknown"):
    """
    流式处理函数，分离思考过程和回复内容
    Args:
        prompt: 用户输入文本
        model: 使用的模型名称
        speaker_id: 说话人标识
    Yields:
        dict: 包含类型和内容的事件对象，格式为:
            {'type': 'thinking', 'content': '...'} 或
            {'type': 'response', 'content': '...'}
    """
    global local_time, formatted_local

    last_time = local_time
    local_time = datetime.now()
    formatted_local = local_time.strftime("%Y年%m月%d日%H时%M分%S秒")

    # 系统提示
    system_prompt = [
        {"role": "system", "content": f"你是超强的人工智能助手，你会灵活的切换钟离或者温迪的角色，你正在和 {speaker_id} 对话。默认助手角色是钟离。"},
        {"role": "system", "content": "使用自然对话的说话方式，只输出中文文字和标点，不输出阿拉伯数字和特殊符号。"},
        {"role": "system", "content": "请标注说话人的身份，说话格式是'[[/speaker_start]说话人[/speaker_end]]说话内容\n[/say_end]'，注意一定要添加句子结尾标识符。"},
        {"role": "system", "content": f"示例'[[/speaker_start]钟离[/speaker_end]]你好， {speaker_id} 。\n[/say_end]'"},
        {"role": "system", "content": "注意说话要自然，符合说话的习惯，简短回复，不要过分重复。注意用户语音输入可能有文字识别错误，尽量理解真实含义。"},
        {"role": "system", "content": "如果用户输入无意义的内容，你应该保持语音沉默。只回复 None。"},
        {"role": "system", "content": "识别到用户输入内容不是在和你说话，与你无关时，你应该保持语音沉默。比如没有喊你的名字时只回复 None。"},
    ]

    # 环境提示（时间）
    if last_time is None or (local_time - last_time) > timedelta(minutes=10):
        environment_prompt = [{
            "role": "system",
            "content": f"当前时间是 {formatted_local}，请根据时间进行适当的回应。"
        }]
        last_time = local_time
    else:
        environment_prompt = []

    # 构造完整消息
    messages = history_manager.get_messages_for_model()
    full_messages = system_prompt + messages + environment_prompt + [{"role": "user", "content": prompt}]

    think_enable = model == "deepseek-r1"

    # 在线程池中执行同步的流式请求
    response_stream = ollama.chat(
        model=model,
        messages=full_messages,
        think=think_enable,
        stream=True
    )

    # 初始化缓冲区
    thinking_buffer = ""
    response_buffer = ""
    thinking_started = False
    thinking_ended = False

    try:
        # 处理流式响应
        for chunk in response_stream:
            # print(f"chunk:{chunk}")
            if hasattr(chunk, 'message'):
                # 处理思考过程
                if chunk.message.thinking:
                    thinking_buffer += chunk.message.thinking
                    if not thinking_started:
                        yield {
                            'type': 'thinking',
                            'content': '[思考过程] ' + chunk.message.thinking
                        }
                        thinking_started = True
                    else:
                        yield {
                            'type': 'thinking',
                            'content': chunk.message.thinking
                        }

                # 处理回复内容
                if chunk.message.content:
                    response_buffer += chunk.message.content
                    sentences = segmenter.push(chunk.message.content)
                    for sentence in sentences:
                        yield {'type': 'response', 'content': sentence}

            # 检查是否中断
            from tts_playback import stop_playback_flag
            if stop_playback_flag.is_set():
                print("[LLM 中断操作] 用户开始说话，跳过当前LLM生成")
                break

        # 输出剩余内容
        if segmenter.buffer:
            yield {'type': 'response', 'content': segmenter.buffer}
            segmenter.buffer = ""

        # 结束思考
        if thinking_started and not thinking_ended:
            yield {
                'type': 'thinking',
                'content': '\n[思考过程结束]'
            }
            thinking_ended = True

    except Exception as e:
        yield {'type': 'error', 'content': f"流式处理异常: {str(e)}"}
    finally:
        # 保存用户输入和模型输出到历史
        history_manager.add_message("user", prompt)
        history_manager.add_message("assistant", response_buffer.strip())
        history_manager.maybe_compress_history()
        history_manager.save_to_file()  # 每次对话后自动保存


def handle_response_event(event: Dict[str, Any], speaker: str, tts_handler):
    """处理响应事件"""
    print(f"event: {event}")
    if event['type'] == 'thinking':
        print(event['content'], end='', flush=True)
    elif event['type'] == 'response':
        print(f"[回复内容] {event['content']}")
        response_speaker = "unknown"
        if event['content'].get('speaker') is not None:
            response_speaker = event['content']['speaker']

        print("response_speaker")
        print(response_speaker)
        if tts_handler:
            tts_handler(event['content']['content'],response_speaker)
    elif event['type'] == 'error':
        print(f"[错误] {event['content']}")


def chat_handle(user_prompt: str, speaker_name: str, tts_handler):
    """主对话处理函数"""
    for event in stream_chat(user_prompt, speaker_id=speaker_name):
        handle_response_event(event, speaker_name, tts_handler)