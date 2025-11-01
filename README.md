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

---

## 📦 安装与运行

### 1️⃣ 克隆仓库并使用 Conda 创建环境：

```bash
git clone https://github.com/gengyuchao/speech_to_speech.git
cd speech_to_speech
git checkout dev/high_quality_main
conda env create -f environment.yml
conda activate speech_assistant
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

- 下载 Whisper 模型（默认为 `large-v3`）：
```bash
# 默认由 Whisper 自动下载，也可以手动指定路径
```

- 下载 Silero VAD 模型（会自动加载）：

```bash
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
```

- Index-TTS 需要你自行配置模型目录和配置文件。

### 3️⃣ 配置文件说明

编辑 `config.yaml` 来设置 TTS 参数、缓存路径等：

```yaml
tts_model_dir: ./models/index-tts
tts_cfg_path: ./models/index-tts/config.yaml
cache_dir: ./cache
speaker_voices:
  unknown: ./voices/default.wav
  玉超: ./voices/yuchao.wav
  钟离: ./voices/zhongli.wav
  温迪: ./voices/wendi.wav
vad:
  sensitivity: 0.6
silence_detection:
  silence_threshold: -50
  min_silence_len: 1000
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
├── main.py                 # 主程序入口
├── ollama_stream.py        # LLM 对话逻辑与流式输出处理
├── audio_manager.py        # 音频采集、识别、打断控制
├── sentence_segmenter.py   # 中文句子分割 + 说话人结构解析
├── text_cleaner.py         # 文本清理规则（去除特殊符号）
├── tts_queue.py            # TTS 队列管理与播放线程
├── vad_controller.py       # VAD 控制器，支持动态调整敏感度
├── config.yaml             # 系统配置文件
├── environment.yml         # Conda 环境配置文件
└── requirements.txt        # Python 依赖项（供参考）
```

---

## 🛠️ 扩展建议

- ✅ 添加更多角色音色支持（如：女声、男声、不同年龄段）
- ✅ 支持多语言模型切换
- ✅ 增加 Web UI 界面或 Telegram Bot 接口
- ✅ 支持语音转文字后自动翻译成英文等

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

MIT License. See `LICENSE` for more information.

--- 

> 💡 **提示**：建议配合耳机/音响使用，以获得更好的语音体验。