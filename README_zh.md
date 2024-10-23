
# 项目简介 🌟

[English Version](./README.md)  

[中文版本](./README_zh.md)

这个项目旨在实现**语音到语音**的转换，提供一个流畅且自然的对话体验。核心技术包括：

- **Whisper.cpp**：用于高精度的语音识别，能够捕捉和理解用户的语音输入。
- **大型语言模型 (LLM)**：负责智能化的对话生成，能够理解上下文并提供相关的回答和建议，提升交互的自然性和智能化水平。
- **Edge_TTS 或 EmotiVoice**：用于将生成的文本转换为语音，支持多种音色和语调，以确保输出的语音听起来自然且富有表现力。

### 功能特点 ✨:
- **实时交互**：支持即时的语音对话，让用户体验顺畅无缝的交流。
- **多样化语音输出**：提供多种语音选择，适应不同场景和用户偏好。
- **高准确度**：结合先进的语音识别和自然语言处理技术，确保用户输入的准确理解和有效反馈。
- **本地化解决方案**：项目支持在本地运行，确保用户数据的隐私和安全。

无论是用于个人助手、教育辅导还是社交应用，这个项目都能提供智能且自然的语音交互体验。 🎤🤖


# 安装与设置指南 📦

## 如何安装组件

### 1. 安装带有 CUDA 的 Whisper.cpp
要设置带有 CUDA 支持的 Whisper.cpp，请导航到该目录并使用以下命令编译源代码：

```bash
cd components/whisper.cpp
GGML_CUDA=1 make -j 8 
```

### 2. 安装 EmotiVoice
有关 EmotiVoice 安装的详细信息，请参考以下文档中的指南：

```
components/EmotiVoice/README_小白安装教程.md
```

## 启动所有服务器

安装完必要组件后，可以按如下方式启动服务器：

### 1. 启动 Whisper 服务器
导航到 Whisper.cpp 目录，并使用指定的模型和语言启动服务器：

```bash
cd components/whisper.cpp
./server -m ./models/ggml-large-v3-q5_0.bin -l zh
```

### 2. 启动 EmotiVoice 服务器
在一个新的终端中，导航到 EmotiVoice 目录并启动服务器：

```bash
cd components/EmotiVoice
uvicorn openaiapi:app --reload --port 6006
```

### 3. 运行主应用程序
最后，执行主应用程序以启动您的项目：

```bash
python3 main.py
```

## 注意事项
- 确保在指定的适当目录中执行所有命令。
- 保持终端开启并运行服务器，以维持功能正常。
