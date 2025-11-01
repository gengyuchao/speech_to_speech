#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一配置管理模块
负责加载、管理和提供系统所有配置
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigManager:
    """统一配置管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._config = {}
            self._load_config()
            self._initialized = True
    
    def _load_config(self):
        """加载配置文件"""
        try:
            config_path = "config.yaml"
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                # 如果配置文件不存在，创建默认配置
                self._config = self._create_default_config()
        except Exception as e:
            print(f"警告: 无法加载配置文件: {e}")
            self._config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        return {
            "tts": {
                "model_dir": "resources/checkpoints",
                "cfg_path": "resources/checkpoints/config.yaml",
                "cache_dir": "./cache",
                "kwargs": {
                    "do_sample": True,
                    "top_p": 0.8,
                    "top_k": 30,
                    "temperature": 1.0,
                    "length_penalty": 0.0,
                    "num_beams": 3,
                    "repetition_penalty": 10.0,
                    "max_mel_tokens": 600
                }
            },
            "speaker_voices": {
                "钟离": "resources/voice/钟离2.wav",
                "温迪": "resources/voice/温迪.mp3",
                "可莉": "resources/voice/可莉3.mp3",
                "胡桃": "resources/voice/胡桃.mp3",
                "玉超": "./voices/yuchao.wav",
                "unknown": "resources/voice/钟离2.wav"
            },
            "vad": {
                "sensitivity": 0.6,
                "play_sensitivity_factor": 0.2
            },
            "asr": {
                "model_path": "resources/Belle-whisper-large-v3-turbo-zh",
                "device": None
            },
            "asr_prompt": "这是钟离、温迪和玉超在进行的人工智能方面的技术讨论，其中包括 whisper 和 LLM 模型的内容。输出需要带标点符号。",
            "silence_detection": {
                "silence_threshold": -50,
                "min_silence_len": 1000
            },
            "worker_counter_start": 1,
            "ollama": {
                "model": "gemma3:27b",
                "max_history": 30,
                "compress_interval": 20
            },
            "audio": {
                "format": "paInt16",
                "channels": 1,
                "rate": 16000,
                "chunk": 512,
                "silence_frame_threshold": 20
            },
            "audio_similarity": {
                "similarity_threshold": 0.85,
                "silence_threshold": 0.01,
                "silence_ratio_threshold": 0.95,
                "fingerprint_size": 1024
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/system.log"
            },
            "ai_prompts": {
                "system_role": "你是超强的人工智能助手，你会灵活的切换钟离、温迪、胡桃、或者可莉的角色，你正在和 {speaker_id} 对话。默认助手角色是钟离。",
                "speaking_format": "使用自然对话的说话方式，只输出中文文字和标点，不输出阿拉伯数字和特殊符号。",
                "speaker_format": "请标注说话人的身份，说话格式是'[[/speaker_start]说话人[/speaker_end]]说话内容\n[/say_end]'，注意一定要添加句子结尾标识符。",
                "example": "示例'[[/speaker_start]钟离[/speaker_end]]你好， {speaker_id} 。\n[/say_end]'",
                "natural_response": "注意说话要自然，符合说话的习惯，简短回复，不要过分重复。注意用户语音输入可能有文字识别错误，尽量理解真实含义。",
                "silence_if_irrelevant": "如果用户输入无意义的内容，你应该保持语音沉默。只回复 None。",
                "silence_if_not_spoken_to": "识别到用户输入内容不是在和你说话，与你无关时，你应该保持语音沉默。比如没有喊你的名字时只回复 None。",
                "time_context": "当前时间是 {current_time}，请根据时间进行适当的回应。"
            },
            "tts_config": {
                "max_mel_tokens": 600,
                "do_sample": True,
                "top_p": 0.8,
                "top_k": 30,
                "temperature": 1.0,
                "length_penalty": 0.0,
                "num_beams": 3,
                "repetition_penalty": 10.0
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        keys = key.split('.')
        config = self._config
        
        # 导航到要设置的配置项的父级
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()

# 创建全局配置管理器实例
config_manager = ConfigManager()

# 为了向后兼容，也导出旧的配置访问方式
def get_config() -> Dict[str, Any]:
    """获取完整配置字典（向后兼容）"""
    return config_manager.get_all()

def get_config_value(key: str, default: Any = None) -> Any:
    """获取特定配置值（向后兼容）"""
    return config_manager.get(key, default)

if __name__ == "__main__":
    # 测试配置管理器
    print("配置管理器测试:")
    print(f"TTS模型路径: {config_manager.get('tts.model_dir')}")
    print(f"默认角色: {config_manager.get('speaker_voices.unknown')}")
    print(f"ASR提示词: {config_manager.get('asr_prompt')}")