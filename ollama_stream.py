#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ollama_stream.py

"""
LLM处理模块
"""
import os
import json
import time
import ollama
from concurrent.futures import ThreadPoolExecutor
from sentence_segmenter import SentenceSegmenter
from datetime import datetime, timedelta
from logger_config import system_logger
from config_manager import config_manager

# 全局工具初始化
segmenter = SentenceSegmenter()
executor = ThreadPoolExecutor(max_workers=1)
default_model = config_manager.get('ollama.model')
history_max = config_manager.get('ollama.max_history')
history_compress_interval = config_manager.get('ollama.compress_interval')

class ChatHistoryManager:
    def __init__(self, model: str = "deepseek-r1", max_history: int = history_max, compress_interval: int = history_compress_interval):
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
        
    def summarize_earliest(self, m: int = 5) -> None:
        """用模型总结最早 m 条历史记录"""
        if len(self.history) < m:
            return
            
        to_summarize = self.history[:-m]
        summary_prompt = (
            "请对以下对话历史进行简洁的总结，保留关键信息和上下文关系。\n\n"
            + "\n".join([f"{msg['role']}: {msg['content']}" for msg in to_summarize])
        )
        
        try:
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": summary_prompt}])
            summary = response["message"]["content"]
            # 替换掉前 m 条为总结
            self.history = [{"role": "system", "content": f"[历史摘要] {summary}"}] + self.history[-m:]
            
        except Exception as e:
            system_logger.error("历史总结失败: {}".format(e))
            
    def maybe_compress_history(self):
        """判断是否需要压缩历史"""
        if (self.total_turns >= self.max_history) and (self.total_turns % self.compress_interval == 0):
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
            
        system_logger.info("[+] 已保存历史到 {}".format(filepath))
        
    def load_from_file(self, filepath: str = "history.json"):
        """从文件加载历史记录"""
        if not os.path.exists(filepath):
            system_logger.info("[!] 文件 {} 不存在，跳过加载".format(filepath))
            return
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.total_turns = data.get("total_turns", 0)
            self.history = data.get("history", [])
            
            system_logger.info("[+] 已从 {} 加载历史记录".format(filepath))
            
        except Exception as e:
            system_logger.error("[!] 加载历史出错: {}".format(e))

# 全局历史管理器实例
history_manager = ChatHistoryManager(model=default_model, max_history=history_max, compress_interval=history_compress_interval)
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
        {"role": "system", "content": config_manager.get('ai_prompts.system_role').format(speaker_id=speaker_id)},
        {"role": "system", "content": config_manager.get('ai_prompts.speaking_format')},
        {"role": "system", "content": config_manager.get('ai_prompts.speaker_format')},
        {"role": "system", "content": config_manager.get('ai_prompts.example').format(speaker_id=speaker_id)},
        {"role": "system", "content": config_manager.get('ai_prompts.natural_response')},
        {"role": "system", "content": config_manager.get('ai_prompts.silence_if_irrelevant')},
        {"role": "system", "content": config_manager.get('ai_prompts.silence_if_not_spoken_to')},
    ]
    
    # 环境提示（时间）
    if last_time is None or (local_time - last_time) > timedelta(minutes=10):
        environment_prompt = [{
            "role": "system",
            "content": config_manager.get('ai_prompts.time_context').format(current_time=formatted_local)
        }]
        last_time = local_time
    else:
        environment_prompt = []
        
    # 构造完整消息
    messages = history_manager.get_messages_for_model()
    full_messages = system_prompt + messages + environment_prompt + [{"role": "user", "content": prompt}]
    
    think_enable = model == "deepseek-r1"
    
    # 记录LLM开始时间
    llm_start_time = time.time()
    token_count = 0  # 记录生成的token数量
    
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
                    token_count += len(chunk.message.content.split())  # 简单的token计数
                    sentences = segmenter.push(chunk.message.content)
                    for sentence in sentences:
                        yield {'type': 'response', 'content': sentence}
                        
            # 检查是否中断
            from tts_playback import stop_playback_flag
            if stop_playback_flag.is_set():
                system_logger.info("[LLM 中断操作] 用户开始说话，跳过当前LLM生成")
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
            
        # 计算LLM总耗时
        llm_total_time = time.time() - llm_start_time
        # 计算处理速度
        speed_info = f"\n[LLM生成完成] 耗时: {llm_total_time:.2f}秒"
        if token_count > 0:
            tokens_per_second = token_count / llm_total_time if llm_total_time > 0 else 0
            speed_info += f", 速度: {tokens_per_second:.1f} tokens/秒"
        
        system_logger.info(speed_info)
        # 也可以通过yield返回给前端显示
        # yield {'type': 'info', 'content': speed_info}
            
    except Exception as e:
        # 记录异常时的耗时
        llm_total_time = time.time() - llm_start_time
        system_logger.error(f"[LLM异常] 耗时: {llm_total_time:.2f}秒, 错误: {str(e)}")
        yield {'type': 'error', 'content': "流式处理异常: {}".format(str(e))}
    finally:
        # 保存用户输入和模型输出到历史
        history_manager.add_message("user", prompt)
        history_manager.add_message("assistant", response_buffer.strip())
        history_manager.maybe_compress_history()
        history_manager.save_to_file()  # 每次对话后自动保存

def handle_response_event(event: dict, speaker: str, tts_handler):
    """处理响应事件"""
    system_logger.debug("event: {}".format(event))
    
    if event['type'] == 'thinking':
        print(event['content'], end='', flush=True)
    elif event['type'] == 'response':
        print("[回复内容] {}".format(event['content']))
        response_speaker = "unknown"
        if event['content'].get('speaker') is not None:
            response_speaker = event['content']['speaker']

            if tts_handler:
                tts_handler(event['content']['content'], response_speaker)

        system_logger.debug("response_speaker")
        system_logger.debug(response_speaker)

    elif event['type'] == 'error':
        print("[错误] {}".format(event['content']))

def chat_handle(user_prompt: str, speaker_name: str, tts_handler):
    """主对话处理函数"""
    for event in stream_chat(user_prompt, speaker_id=speaker_name):
        handle_response_event(event, speaker_name, tts_handler)