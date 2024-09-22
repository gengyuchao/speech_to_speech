# speech_to_speech
speech to speech, use whisper.cpp to hear, LLM to think and, edge_tts or EmotiVoice to say.

## How to install components

Install Whisper.cpp with CUDA.
```
cd components/whisper.cpp
GGML_CUDA=1 make -j 8 
```

Install EmotiVoice.

components/EmotiVoice/README_小白安装教程.md


## Start all servers

```
cd components/whisper.cpp
./server -m  ./models/ggml-large-v3-q5_0.bin -l zh
```

```
cd components/EmotiVoice
uvicorn openaiapi:app --reload --port 6006
```

```
python3 main.py
```

