#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vad_controller.py

"""
VAD控制器模块
"""
import threading
import queue
from logger_config import system_logger
from config_manager import config_manager

# 从配置文件导入参数
# 使用统一配置管理器

class VadController:
    def __init__(self):
        self._sensitivity = config_manager.get('vad.sensitivity')
        self._is_playing = False
        self._play_sensitivity_factor = config_manager.get('vad.play_sensitivity_factor')
        self._lock = threading.RLock()  # 使用可重入锁
        self._command_queue = queue.Queue()  # 命令队列
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        
    def _worker(self):
        """后台工作线程处理命令"""
        while self._running:
            try:
                command, value = self._command_queue.get(timeout=0.1)
                
                if command == 'set_playing':
                    self._set_playing_internal(value)
                elif command == 'set_sensitivity':
                    self._set_sensitivity_internal(value)
                    
            except queue.Empty:
                continue
            except Exception as e:
                system_logger.error("VAD 控制器工作线程错误: {}".format(e))
                
    def _set_playing_internal(self, playing: bool):
        """内部设置播放状态（线程安全）"""
        with self._lock:
            old_state = self._is_playing
            self._is_playing = playing
            
            if old_state != playing:
                status = "播放中" if playing else "播放结束"
                # system_logger.debug("[VAD状态] {} - 当前阈值: {:.3f}".format(status, self.get_threshold()))
                
    def _set_sensitivity_internal(self, value: float):
        """内部设置敏感度（线程安全）"""
        with self._lock:
            self._sensitivity = max(0.0, min(1.0, value))
            system_logger.info("基础VAD敏感度设置为: {}".format(self._sensitivity))
            
    def set_playing(self, playing: bool):
        """异步设置播放状态"""
        try:
            self._command_queue.put(('set_playing', playing), block=False)
        except queue.Full:
            system_logger.warning("VAD 控制命令队列已满")
            
    def set_sensitivity(self, value: float):
        """异步设置敏感度"""
        try:
            self._command_queue.put(('set_sensitivity', value), block=False)
        except queue.Full:
            system_logger.warning("VAD 控制命令队列已满")
            
    def get_threshold(self):
        with self._lock:
            if self._is_playing:
                return min(0.95, self._sensitivity + self._play_sensitivity_factor)
            return self._sensitivity
            
    def stop(self):
        """停止控制器"""
        self._running = False