#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sentence_segmenter.py

"""
句子分割模块
"""
import torch
import functools
from ltp import StnSplit
from logger_config import system_logger

# # Monkey patch torch.load to support PyTorch 2.6+
# old_torch_load = torch.load
# torch.load = functools.partial(old_torch_load, weights_only=False)

class SentenceSegmenter:
    def __init__(self):
        self.splitter = StnSplit()
        self.buffer = ""  # 缓冲未闭合的普通文本
        self.speech_buffer = ""  # 缓冲未闭合的说话内容
        
        # 说话人识别标识符
        self.SPEAKER_START = '[[/speaker_start]'
        self.SPEAKER_END = '[/speaker_end]]'
        self.SPEECH_END = '[/say_end]'
        
        # 状态机状态
        self.state = 'idle'  # idle | speaker | content
        self.speaker_buffer = ""
        self.content_buffer = ""
        
    def push(self, new_text):
        """接收新的文本片段，返回新闭合的句子列表（兼容说话人识别）"""
        results = []
        
        # 处理说话人结构
        speech_results = self._process_speech_stream(new_text)
        results.extend(speech_results)
        
        return results
        
    def _process_speech_stream(self, new_text):
        """流式解析说话人结构"""
        results = []
        self.speech_buffer += new_text
        
        # system_logger.debug("DEBUG: new_text='{}'".format(new_text))
        # system_logger.debug("DEBUG: state='{}', speech_buffer='{}'".format(self.state, self.speech_buffer))
        
        while True:
            if self.state == 'idle':
                # 查找说话人开始标记
                start_idx = self.speech_buffer.find(self.SPEAKER_START)
                
                # system_logger.debug("DEBUG: idle状态查找开始标记'{}'，位置={}".format(self.SPEAKER_START, start_idx))
                
                if start_idx != -1:
                    # 找到了说话人开始标记
                    # 开始标记之前的内容作为普通文本返回
                    if start_idx > 0:
                        prefix_text = self.speech_buffer[:start_idx]
                        if prefix_text:
                            results.append({
                                'type': 'text',
                                'content': prefix_text
                            })
                            system_logger.debug("DEBUG: 返回前缀文本: '{}'".format(prefix_text))
                            
                    self.speech_buffer = self.speech_buffer[start_idx + len(self.SPEAKER_START):]
                    self.state = 'speaker'
                    self.speaker_buffer = ""
                    self.speaker_buffer += self.speech_buffer
                    
                    # system_logger.debug("DEBUG: 找到开始标记，切换到speaker状态, 新buffer='{}'".format(self.speech_buffer))
                else:
                    # 没找到开始标记，保留buffer等待更多数据
                    # system_logger.debug("DEBUG: 未找到开始标记，保持idle状态，buffer='{}'".format(self.speech_buffer))
                    break
                    
            elif self.state == 'speaker':
                # 查找说话人结束标记
                end_idx = self.speech_buffer.find(self.SPEAKER_END)
                
                # system_logger.debug("DEBUG: speaker状态查找结束标记'{}'，位置={}".format(self.SPEAKER_END, end_idx))
                # system_logger.debug("DEBUG: self.speech_buffer='{}' end_idx={}".format(self.speech_buffer, end_idx))
                
                if end_idx != -1:
                    # 找到了说话人结束标记
                    speaker_name = self.speech_buffer[:end_idx]
                    self.speaker_buffer = speaker_name
                    self.speech_buffer = self.speech_buffer[end_idx + len(self.SPEAKER_END):]
                    self.state = 'content'
                    self.content_buffer = ""
                    self.content_buffer += self.speech_buffer
                    
                    # system_logger.debug("DEBUG: 找到结束标记，speaker='{}', 新buffer='{}'".format(speaker_name, self.speech_buffer))
                    
                    sentence_results = self._process_sentences(
                        self.content_buffer,
                        msg_type='speech',
                        speaker=self.speaker_buffer.strip()
                    )
                    
                    # 添加分句结果到最终返回列表
                    results.extend(sentence_results)
                    break
                else:
                    # 没找到结束标记，累积当前buffer到speaker_buffer
                    self.speaker_buffer += new_text
                    
                    # system_logger.debug("DEBUG: 未找到结束标记，累积到speaker_buffer='{}'".format(self.speaker_buffer))
                    break
                    
            elif self.state == 'content':
                # 查找说话结束标记
                end_idx = self.speech_buffer.find(self.SPEECH_END)
                
                # system_logger.debug("DEBUG: self.speech_buffer'{}'".format(self.speech_buffer))
                # system_logger.debug("DEBUG: content状态查找结束标记'{}'，位置={}".format(self.SPEECH_END, end_idx))
                
                if end_idx != -1:
                    # 找到了说话结束标记，这是一个完整结构
                    content_text = self.speech_buffer[:end_idx]
                    self.content_buffer = content_text
                    
                    # 调用 _process_sentences 处理分句
                    sentence_results = self._process_sentences(
                        new_text,
                        msg_type='speech',
                        speaker=self.speaker_buffer.strip()
                    )
                    
                    # 添加分句结果到最终返回列表
                    results.extend(sentence_results)
                    
                    # 处理剩余部分
                    self.speech_buffer = self.speech_buffer[end_idx + len(self.SPEECH_END):]
                    self.state = 'idle'
                    self.speaker_buffer = ""
                    self.content_buffer = ""
                    
                    # system_logger.debug("DEBUG: 完成结构处理，剩余buffer='{}'".format(self.speech_buffer))
                    
                    # 继续处理buffer中可能存在的下一个结构
                else:
                    # 没找到结束标记，累积当前buffer到content_buffer
                    self.content_buffer += new_text
                    
                    # 调用 _process_sentences 处理分句
                    sentence_results = self._process_sentences(
                        new_text,
                        msg_type='speech',
                        speaker=self.speaker_buffer.strip()
                    )
                    
                    # 添加分句结果到最终返回列表
                    results.extend(sentence_results)
                    
                    # system_logger.debug("DEBUG: 未找到结束标记，累积到content_buffer='{}'".format(self.content_buffer))
                    break
                    
        # system_logger.debug("DEBUG: 最终结果={}".format(results))
        return results
        
    def _process_sentences(self, new_text, msg_type, speaker=None):
        """处理普通句子分割逻辑，支持 type 和 speaker"""
        results = []
        self.buffer += new_text
        
        # 如果有换行符，优先处理第一段
        if '\n' in self.buffer:
            parts = self.buffer.split('\n', 1)
            first_line = parts[0].strip()
            
            if first_line:
                results.append({
                    'type': msg_type,
                    'speaker': speaker,
                    'content': first_line
                })
                
            self.buffer = parts[1] if len(parts) > 1 else ""
            
        end_idx = self.buffer.find(self.SPEECH_END)
        
        # system_logger.debug("self.buffer:{}:: end_idx:{}".format(self.buffer, end_idx))
        
        if end_idx != -1:
            # 找到了说话结束标记，这是一个完整结构
            content_text = self.buffer[:end_idx]
            
            if content_text != "":
                results.append({
                    'type': msg_type,
                    'speaker': speaker,
                    'content': content_text
                })
                
            self.buffer = ""
            
        # 检查是否满足最小长度要求
        if len(self.buffer) < 15:
            return results
            
        # 使用 StnSplit 分割句子
        sentences = self.splitter.split(self.buffer)
        merged = []
        total_len = 0
        
        for sent in sentences:
            merged.append(sent)
            total_len += len(sent)
            
            if total_len >= 15 and self._is_closed_sentence(sent):
                full_sentence = ''.join(merged)
                
                results.append({
                    'type': msg_type,
                    'speaker': speaker,
                    'content': full_sentence
                })
                
                self.buffer = self.buffer[len(full_sentence):]
                break
                
        return results
        
    def _is_closed_sentence(self, sentence):
        sentence = sentence.strip()
        return sentence.endswith(("。", "！", "？", "；", "…", "。”", "！”", "？”", "\n"))
        
    def flush(self, msg_type, speaker=None):
        """强制输出剩余 buffer 中的内容（用于结束时）"""
        results = []
        
        # 输出未闭合的说话内容
        if self.buffer.strip():
            results.append({
                'type': msg_type,
                'speaker': speaker,
                'content': self.buffer.replace(self.SPEECH_END,"").strip()
            })
            
            self.buffer = ""
            
        self.speech_buffer = ""
        return results

if __name__ == "__main__":
    segmenter = SentenceSegmenter()
    
    text = "[[/speaker_start]钟离[/speaker_end]]如此，修复错误是好事。\n[/say_end]\n"
    text += "[[/speaker_start]温迪[/speaker_end]]哦？看来你终于意识到需要我的声音了。真是令人惊喜啊。\n[/say_end]"
    
    # 两个字两个字地切分
    chunks = [text[i:i+3] for i in range(0, len(text), 3)]
    
    for chunk in chunks:
        result = segmenter.push(chunk)
        if result:
            for item in result:
                print(item)