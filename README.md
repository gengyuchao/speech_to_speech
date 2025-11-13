# AI 对话助手（语音交互系统）speech to speech

这是一个基于本地运行的大语言模型（LLM）、语音识别、语音合成和语音活动检测技术构建的 **智能对话助手**，支持语音输入与文本输入，并能自动处理语音打断、角色切换等复杂功能。

---

## 🔧 技术栈

| 模块 | 工具/框架 | 用途 |
|------|-----------|------|
| 🤖 大语言模型 | [Ollama](https://ollama.com/) + [Gemma3:27b](https://ollama.com/library/gemma3) / [DeepSeek-R1](https://www.deepseek.com/) | 对话生成、思考过程处理 |
| 🎤 语音识别 | [Whisper](https://github.com/openai/whisper) | 将语音转换为文本 |
| 🧠 语音活动检测 (VAD) | [Silero VAD](https://github.com/snakers4/silero-vad) | 检测用户说话开始与结束 |
| 🗣️ 语音合成 | [Index-TTS](https://github.com/index-tts/index-tts) | 将文字转换为语音输出 |
| 📖 中文句子分割 | [LTP](https://github.com/HIT-SCIR/ltp) | 文本分句处理，支持说话人结构识别 |

---

## 🌟 功能亮点

- ✅ **多模态交互**：支持键盘输入 + 语音输入
- ✅ **智能打断机制**：当用户说话时自动中断当前 TTS 播放
- ✅ **角色切换能力**：可灵活切换"钟离"、"温迪"或自定义角色进行回复
- ✅ **本地运行模型**：所有核心组件均可在本地运行，无需联网
- ✅ **语音相似度检测**：避免播放内容被误识别为用户输入
- ✅ **对话历史管理与压缩**：自动保存和总结历史记录
- ✅ **可调敏感度控制**：通过命令 `v0.3` 调整 VAD 敏感度
- ✅ **多角色音色支持**：支持多种角色音色，包括钟离、温迪、可莉、胡桃等

---

## 📦 安装与运行

### 1️⃣ 克隆仓库并使用 Conda 创建环境：

```bash
git clone https://github.com/gengyuchao/speech_to_speech.git
cd speech_to_speech
conda env create -f environment.yml
conda activate realtime_ai
```

```bash
sudo apt install libaio-dev
sudo apt-get install portaudio19-dev ffmpeg
```

### 2️⃣ 下载模型文件

- 使用 Ollama 安装 Gemma3 或 DeepSeek 模型：
```bash
ollama pull gemma3:27b
ollama pull deepseek-r1
```

- 下载 faster_whisper 模型（默认为 `large-v3-turbo`）：
```bash
# 默认由 faster_whisper 自动下载，也可以手动指定路径
```

- 下载 Silero VAD 模型（会自动加载）：

```bash
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
# 默认自动下载，也可以手动指定路径
```

- Index-TTS 需要你自行配置模型目录和配置文件。

### 3️⃣ 配置文件说明

编辑 `config.yaml` 来设置 TTS 参数、缓存路径等：

```yaml
tts:
  model_dir: "resources/checkpoints"
  cfg_path: "resources/checkpoints/config.yaml"
  cache_dir: "./cache"
  kwargs:
    do_sample: true
    top_p: 0.8
    top_k: 30
    temperature: 1.0
    length_penalty: 0.0
    num_beams: 3
    repetition_penalty: 10.0
    max_mel_tokens: 600

speaker_voices:
  "钟离": "resources/voice/钟离2.wav"
  "温迪": "resources/voice/温迪.mp3"
  "可莉": "resources/voice/可莉3.mp3"
  "胡桃": "resources/voice/胡桃.mp3"
  "玉超": "./voices/yuchao.wav"
  "unknown": "resources/voice/钟离2.wav"

vad:
  sensitivity: 0.6
  play_sensitivity_factor: 0.2

asr:
  model_path: "resources/Belle-whisper-large-v3-turbo-zh"
  device: null  # null表示自动选择设备

# ASR提示词配置
asr_prompt: "这是钟离、温迪和玉超在进行的人工智能方面的技术讨论，其中包括 whisper 和 LLM 模型的内容。输出需要带标点符号。"

silence_detection:
  silence_threshold: -50
  min_silence_len: 1000

worker_counter_start: 1

ollama:
  model: "gemma3:27b"
  # model: "qwen3:32b"
  # model: "deepseek-r1:latest"
  max_history: 30
  compress_interval: 20

audio:
  format: "paInt16"
  channels: 1
  rate: 16000
  chunk: 512
  silence_frame_threshold: 20

audio_similarity:
  similarity_threshold: 0.85
  silence_threshold: 0.01
  silence_ratio_threshold: 0.95
  fingerprint_size: 1024

logging:
  level: "INFO"
  file: "./logs/system.log"

# AI提示词配置
ai_prompts:
  system_role: "你是超强的人工智能助手，你会灵活的切换钟离、温迪、胡桃、或者可莉的角色，你正在和 {speaker_id} 对话。默认助手角色是钟离。"
  speaking_format: "使用自然对话的说话方式，只输出中文文字和标点，不输出阿拉伯数字和特殊符号。"
  speaker_format: "请标注说话人的身份，说话格式是'[[/speaker_start]说话人[/speaker_end]]说话内容\n[/say_end]'，注意一定要添加句子结尾标识符。"
  example: "示例'[[/speaker_start]钟离[/speaker_end]]你好， {speaker_id} 。\n[/say_end]'"
  natural_response: "注意说话要自然，符合说话的习惯，简短回复，不要过分重复。注意用户语音输入可能有文字识别错误，尽量理解真实含义。"
  silence_if_irrelevant: "如果用户输入无意义的内容，你应该保持语音沉默。只回复 None。"
  silence_if_not_spoken_to: "识别到用户输入内容不是在和你说话，与你无关时，你应该保持语音沉默。比如没有喊你的名字时只回复 None。"
  time_context: "当前时间是 {current_time}，请根据时间进行适当的回应。"

# TTS配置
tts_config:
  max_mel_tokens: 600
  do_sample: true
  top_p: 0.8
  top_k: 30
  temperature: 1.0
  length_penalty: 0.0
  num_beams: 3
  repetition_penalty: 10.0
```

### 4️⃣ 启动程序

```bash
python main.py
```

启动后你将看到提示：
```
语音识别已启动，你可以开始说话...
请输入你的问题（按 q 退出，v+数字 调节敏感度）：
```

---

## 🎮 使用方式

### 输入方式

- 键盘输入：直接输入文字即可。
- 语音输入：对着麦克风说话即可。

### 命令控制

| 命令 | 功能 |
|------|------|
| `q` / `quit` / `exit` | 退出程序 |
| `v0.3` | 设置 VAD 敏感度为 0.3（范围 0~1） |

---

## 🧠 工作流程图

```text
[用户说话]
     ↓
 [语音识别 → Whisper]
     ↓
 [语音活动检测 → Silero VAD]
     ↓
 [打断当前 TTS 播放]
     ↓
 [LLM 处理 → Ollama + 内容结构化]
     ↓
 [文本分句 + 角色识别]
     ↓
 [TTS 合成 → Index-TTS]
     ↓
 [播放语音 → SoundDevice]
```

---

## 📁 目录结构

```bash
.
├── asr.py                  # 语音识别模块
├── audio_manager.py        # 音频采集、识别、打断控制
├── config_manager.py
├── config.yaml             # 系统配置文件
├── environment.yml         # Conda 环境配置文件
├── history.json
├── LICENSE
├── logger_config.py        # 日志配置模块
├── logs
│   └── system.log
├── main.py                 # 主程序入口
├── ollama_stream.py        # LLM 对话逻辑与流式输出处理
├── README.md
├── requirements.txt        # Python 依赖项（供参考）
├── resources
│   ├── Belle-whisper-large-v3-turbo-zh # 中文专用 whisper 模型
│   ├── checkpoints         # index-TTS2 的模型 checkpoints
│   └── voice               # 声音音色资源
├── sentence_segmenter.py   # 中文句子分割 + 说话人结构解析
├── snakers4                # VAD 模型
│   └── silero-vad
├── text_cleaner.py         # 文本清理规则（去除特殊符号）
├── tts_playback.py         # TTS 队列管理与播放线程
└── vad_controller.py       # VAD 控制器，支持动态调整敏感度

```

---

## 🛠️ 扩展建议

- ✅ 增加 Web UI 界面
- ✅ 支持语音转文字后自动翻译成英文等
- ✅ 支持 Deepseek ocr 视觉识别
- ✅ 支持 webrtc 回声消除
- ✅ 支持 MCP 操作指令，实现读取文件，访问网络等功能

## 效果演示

[![视频演示](https://i1.hdslb.com/bfs/archive/f3a94467ff18cf7766683bae647d9cf6d3c60454.jpg@308w_174h)](https://www.bilibili.com/video/BV1fjbMzyE1e)

---

## 🙏 致谢

本项目使用了以下开源工具：

- [Ollama](https://ollama.com/) - 用于运行大型语言模型。
- [Whisper](https://github.com/openai/whisper) - 用于语音识别。
- [Silero VAD](https://github.com/snakers4/silero-vad) - 用于语音活动检测。
- [Index-TTS](https://github.com/index-tts/index-tts) - 用于语音合成。
- [LTP](https://github.com/HIT-SCIR/ltp) - 用于中文句子分割。

---

## 📄 License

MIT License. See `LICENSE` for more information。

---

> 💡 **提示**：建议配合耳机/音响使用，以获得更好的语音体验。

