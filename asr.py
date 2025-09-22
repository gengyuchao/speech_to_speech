# asr.py
import whisper
import wave
import torch
import yaml

# 从配置文件导入参数
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class WhisperASR:
    def __init__(self, model_name="turbo"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_file, language="zh", prompt=config['asr_prompt']):
        result = self.model.transcribe(audio_file, language=language, prompt=prompt)
        return result["text"]


from transformers import pipeline



class TransformersASR:
    def __init__(self, model_path=None, device=None):
        """
        初始化ASR模型
        
        Args:
            model_path: 模型路径，如果为None则使用配置文件中的路径
            device: 设备设置，None表示自动选择，"cuda"表示GPU，"cpu"表示CPU
        """
        # 如果没有指定模型路径，则使用配置文件中的路径
        if model_path is None:
            model_path = config['asr']['model_path']
        
        # 自动选择设备：如果未指定且CUDA可用则使用GPU，否则使用CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # 创建pipeline时指定设备
        self.transcriber = pipeline(
            "automatic-speech-recognition", 
            model=model_path,
            device=device
        )
        
        # 设置强制解码器ID，指定语言和任务
        self.transcriber.model.config.forced_decoder_ids = (
            self.transcriber.tokenizer.get_decoder_prompt_ids(
                language="zh", 
                task="transcribe"
            )
        )

        # 检查模型是否在正确的设备上
        model_device = next(self.transcriber.model.parameters()).device
        print("Model device:", model_device)

    def transcribe(self, audio_file, language="zh", prompt=""):
        """
        转录音频文件
        
        Args:
            audio_file: 音频文件路径
            language: 语言代码
            prompt: 提示文本（注意：某些模型可能不支持）
        """
        try:
            result = self.transcriber(audio_file)
            
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

# 添加到config.yaml中的配置项
# asr:
#   model_path: "resources/Belle-whisper-large-v3-turbo-zh"
#   device: null  # null表示自动选择设备