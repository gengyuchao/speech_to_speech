# Project Overview 🌟

[English Version](./README.md)  

[中文版本](./README_zh.md)

This project aims to achieve **Speech-to-Speech** conversion, providing a seamless and natural conversational experience. The core technologies include:

- **Whisper.cpp**: Utilizes high-precision speech recognition to accurately capture and understand user voice inputs.
- **Large Language Model (LLM)**: Responsible for intelligent dialogue generation, comprehending context, and delivering relevant responses and suggestions, thereby enhancing interaction's naturalness and intelligence.
- **Edge_TTS or EmotiVoice**: Converts the generated text into speech, supporting a variety of tones and intonations to ensure the output sounds natural and expressive.

### Key Features ✨:
- **Real-Time Interaction**: Enables instantaneous voice conversations, allowing users to experience smooth and uninterrupted communication.
- **Diverse Voice Output**: Offers multiple voice options to cater to different scenarios and user preferences.
- **High Accuracy**: Combines advanced speech recognition and natural language processing technologies to ensure precise understanding of user inputs and effective feedback.
- **Localized Solution**: The project supports local execution, ensuring user data privacy and security.

Whether for personal assistants, educational tutoring, or social applications, this project delivers an intelligent and natural voice interaction experience. 🎤🤖



# Installation and Setup Guide 📦

## How to Install Components

### 1. Install Whisper.cpp with CUDA
To set up Whisper.cpp with CUDA support, navigate to the directory and compile the source code using the following commands:

```bash
cd components/whisper.cpp
GGML_CUDA=1 make -j 8 
```

### 2. Install EmotiVoice
For EmotiVoice installation, please refer to the detailed guide provided in the following document:

```
components/EmotiVoice/README_小白安装教程.md
```

## Start All Servers

Once you have installed the necessary components, you can start the servers as follows:

### 1. Start the Whisper Server
Navigate to the Whisper.cpp directory and start the server with the specified model and language:

```bash
cd components/whisper.cpp
./server -m ./models/ggml-large-v3-q5_0.bin -l zh
```

### 2. Start the EmotiVoice Server
In a separate terminal, navigate to the EmotiVoice directory and launch the server:

```bash
cd components/EmotiVoice
uvicorn openaiapi:app --reload --port 6006
```

### 3. Run the Main Application
Finally, execute the main application to start your project:

```bash
python3 main.py
```

## Notes
- Ensure that all commands are executed in the appropriate directories as specified.
- Keep your terminal open and running the servers to maintain functionality.

