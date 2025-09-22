# audio_similarity_detector.py

import numpy as np
from collections import deque
import yaml

# 从配置文件导入参数
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class AudioSimilarityDetector:
    def __init__(self, sample_rate=16000, buffer_duration=2.0):
        """
        音频相似度检测器
        :param sample_rate: 采样率
        :param buffer_duration: 缓冲区时长（秒）
        """
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # 播放音频缓冲区 - 存储最近播放的音频片段
        self.playback_buffer = deque(maxlen=10)  # 存储最近10个片段
        
        # 音频特征参数
        self.fingerprint_size = 1024  # 指纹大小
        self.similarity_threshold = config['audio_similarity']['similarity_threshold']  # 相似度阈值
        
        # 静音检测参数
        self.silence_threshold = config['audio_similarity']['silence_threshold']  # 静音阈值 (归一化幅度)
        self.silence_ratio_threshold = config['audio_similarity']['silence_ratio_threshold']  # 静音比例阈值 (95%以上为静音)
        
    def is_silent_audio(self, audio_data, threshold=None, ratio_threshold=None):
        """
        检查音频是否为静音
        :param audio_data: numpy array, 归一化到 [-1, 1] 的音频数据
        :param threshold: 静音幅度阈值
        :param ratio_threshold: 静音比例阈值
        :return: bool 是否为静音
        """
        if audio_data is None or len(audio_data) == 0:
            return True
            
        if threshold is None:
            threshold = self.silence_threshold
        if ratio_threshold is None:
            ratio_threshold = self.silence_ratio_threshold
            
        # 计算绝对幅度
        abs_audio = np.abs(audio_data)
        
        # 计算静音样本比例
        silent_samples = np.sum(abs_audio < threshold)
        silent_ratio = silent_samples / len(audio_data)
        
        return silent_ratio >= ratio_threshold
        
    def add_playback_audio(self, audio_data):
        """
        添加播放的音频数据
        :param audio_data: numpy array, 归一化到 [-1, 1] 的音频数据
        """
        if audio_data is None or len(audio_data) == 0:
            return
            
        # 排除静音音频
        if self.is_silent_audio(audio_data):
            # print("跳过静音播放音频")
            return
            
        # 添加到缓冲区
        self.playback_buffer.append({
            'audio': audio_data.copy(),
            'timestamp': np.datetime64('now')
        })
        
    def is_similar_to_playback(self, recorded_audio, threshold=None):
        """
        检查录制的音频是否与播放音频相似
        :param recorded_audio: numpy array, 归一化到 [-1, 1] 的录制音频
        :param threshold: 相似度阈值
        :return: (bool, float) 是否相似, 相似度分数
        """
        # 排除静音音频
        if self.is_silent_audio(recorded_audio):
            # print("录制音频为静音，跳过相似度检测")
            return False, 0.0
            
        if len(self.playback_buffer) == 0:
            return False, 0.0
            
        if threshold is None:
            threshold = self.similarity_threshold
            
        max_similarity = 0.0
        is_similar = False
        
        # 检查与所有缓冲的播放音频的相似度
        for playback_item in self.playback_buffer:
            similarity = self._compute_similarity(recorded_audio, playback_item['audio'])
            if similarity > max_similarity:
                max_similarity = similarity
                
            if similarity >= threshold:
                is_similar = True
                break
                
        return is_similar, max_similarity
        
    def _compute_similarity(self, recorded_audio, playback_audio):
        """
        计算两个音频片段的相似度
        支持片段匹配（在长音频中查找短音频）
        """
        if len(recorded_audio) == 0 or len(playback_audio) == 0:
            return 0.0
            
        # 确定哪个是片段，哪个是完整音频
        if len(recorded_audio) > len(playback_audio):
            # recorded 是完整音频，playback 是片段
            long_audio, short_audio = recorded_audio, playback_audio
            is_recorded_longer = True
        else:
            # playback 是完整音频，recorded 是片段
            long_audio, short_audio = playback_audio, recorded_audio
            is_recorded_longer = False
            
        # 如果短音频太短，直接比较
        if len(short_audio) < 32:
            return self._compute_similarity_direct(short_audio, long_audio[:len(short_audio)])
            
        # 使用滑动窗口在长音频中查找最佳匹配
        max_similarity = 0.0
        
        # 窗口大小为短音频的长度，但不超过最大处理长度
        window_size = min(len(short_audio), self.fingerprint_size)
        step_size = max(1, window_size // 4)  # 25% 重叠
        
        # 在长音频中滑动窗口
        for i in range(0, len(long_audio) - window_size + 1, step_size):
            window = long_audio[i:i + window_size]
            
            # 计算当前窗口与短音频的相似度
            similarity = self._compute_similarity_direct(short_audio[:window_size], window)
            
            if similarity > max_similarity:
                max_similarity = similarity
                
            # 如果已经找到很高的相似度，可以提前退出
            if max_similarity > 0.95:
                break
                
        return max_similarity
        
    def _compute_similarity_direct(self, audio1, audio2):
        """
        直接比较两个相同长度的音频片段
        使用多种方法综合判断
        """
        if len(audio1) == 0 or len(audio2) == 0:
            return 0.0
            
        # 确保两个音频长度相同（取较短的）
        min_len = min(len(audio1), len(audio2))
        if min_len == 0:
            return 0.0
            
        a1 = audio1[:min_len]
        a2 = audio2[:min_len]
        
        # 方法1: 互相关相似度
        correlation_sim = self._correlation_similarity_direct(a1, a2)
        
        # 方法2: 能量相似度
        energy_sim = self._energy_similarity_direct(a1, a2)
        
        # 方法3: 频谱相似度
        spectral_sim = self._spectral_similarity_direct(a1, a2)
        
        # 综合相似度
        combined_similarity = 0.5 * correlation_sim + 0.3 * energy_sim + 0.2 * spectral_sim
        
        return min(1.0, max(0.0, combined_similarity))
        
    def _correlation_similarity_direct(self, audio1, audio2):
        """基于互相关的相似度计算（直接比较）"""
        if len(audio1) < 10:
            return 0.0
            
        # 计算互相关
        correlation = np.correlate(audio1, audio2, mode='valid')
        if len(correlation) > 0:
            # 归一化互相关
            normalized_corr = np.max(correlation) / (np.std(audio1) * np.std(audio2) * len(audio1))
            return min(1.0, max(0.0, normalized_corr))
        return 0.0
        
    def _energy_similarity_direct(self, audio1, audio2):
        """基于能量的相似度计算（直接比较）"""
        energy1 = np.mean(audio1 ** 2)
        energy2 = np.mean(audio2 ** 2)
        
        if energy1 == 0 and energy2 == 0:
            return 1.0
        if energy1 == 0 or energy2 == 0:
            return 0.0
            
        # 能量比值相似度
        energy_ratio = min(energy1 / energy2, energy2 / energy1)
        return min(1.0, energy_ratio)
        
    def _spectral_similarity_direct(self, audio1, audio2):
        """基于频谱的相似度计算（直接比较）"""
        if len(audio1) < 32:
            return 0.0
            
        # 计算频谱
        fft1 = np.abs(np.fft.fft(audio1))
        fft2 = np.abs(np.fft.fft(audio2))
        
        # 计算频谱相似度
        norm1 = np.linalg.norm(fft1)
        norm2 = np.linalg.norm(fft2)
        
        if norm1 == 0 and norm2 == 0:
            return 1.0
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # 余弦相似度
        dot_product = np.dot(fft1, fft2)
        similarity = dot_product / (norm1 * norm2)
        return min(1.0, max(0.0, similarity))
        
    def clear_buffer(self):
        """清空播放缓冲区"""
        self.playback_buffer.clear()
        
    def set_silence_threshold(self, threshold):
        """设置静音检测阈值"""
        self.silence_threshold = max(0.0, min(1.0, threshold))
        
    def set_silence_ratio_threshold(self, ratio_threshold):
        """设置静音比例阈值"""
        self.silence_ratio_threshold = max(0.0, min(1.0, ratio_threshold))