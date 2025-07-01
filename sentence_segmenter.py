# sentence_segmenter.py
import torch
import functools
from ltp import StnSplit

# # Monkey patch torch.load to support PyTorch 2.6+
# old_torch_load = torch.load
# torch.load = functools.partial(old_torch_load, weights_only=False)

class SentenceSegmenter:
    def __init__(self):
        self.splitter = StnSplit()
        self.buffer = ""  # 缓冲未闭合的文本

    def push(self, new_text):
        """接收新的文本片段，返回新闭合的句子列表"""
        self.buffer += new_text
        return self._try_split()


    def _try_split(self):
        sentences = []
        text = self.buffer

        # ✅ 条件1：如果总长度小于15，但有换行符，则只处理第一个换行前的部分
        if '\n' in text:
            first_part, sep, rest = text.partition('\n')
            if first_part.strip():
                sentences.append(first_part)
                text = text[len(first_part):]
            self.buffer = rest  # 更新buffer为换行后的内容
        
        if len(text) < 15:
            return sentences  # ✅ 不足15长度时直接返回结果
            
        while True:
            result = self.splitter.split(text)
            if not result:
                break

            # ✅ 如果只分出一个句子，说明后续还可能有内容，先不处理
            if len(result) == 1:
                if not self._is_closed_sentence(result[0]):
                    break  # 可能还没结束，等待更多数据

            merged = []
            current_len = 0
            for i, sent in enumerate(result):
                # print(sent)
                if i == len(result) - 1:
                    if not self._is_closed_sentence(sent):
                        # print("句子还未结束")
                        break  # 可能还没结束，等待更多数据
                merged.append(sent)
                current_len += len(sent)
                if current_len >= 15:
                    head_part = ''.join(merged)
                    sentences.append(head_part)
                    text = text[len(head_part):]
                    break

            break  # 可能还没结束，等待更多数据

        self.buffer = text
        return sentences
        
    def _is_closed_sentence(self, sentence):
        sentence = sentence.strip()
        return sentence.endswith(("。", "！", "？", "；", "…", "。”", "！”", "？”", "\n"))

    def flush(self):
        """强制输出剩余 buffer 中的内容（用于结束时）"""
        remaining = self.buffer.strip()
        self.buffer = ""
        return [remaining] if remaining else []

    
if __name__ == "__main__":
    segmenter = SentenceSegmenter()
    text = "啊，是这样啊！我明白了。声音识别，也就是语音转文本（Speech-to-Text，STT）！\n\n看来你希望在远程使用之前，先让程序能够识别你的声音指令，对吗？\n\n那确实是更基础的问题，优先解决语音识别是正确的。\n\n那么，解决语音识别的问题，也有几种方案：\n\n**1. 使用云端语音识别API：**\n\n*   **选择API：** 谷歌云语音识别 (Google Cloud Speech-to-Text)、亚马逊转录 (Amazon Transcribe)、微软 Azure 语音服务 (Microsoft Azure Speech Services) 都是不错的选择。\n*   **上传音频：** 将录制到的音频上传到云端服务器。\n*   **识别文本：** 使用云端API将音频转换为文本。\n*   **传输文本：** 将识别到的文本传输到本地或远程服务器。\n\n**2. 使用本地语音识别库：**\n\n*   **选择库：** Kaldi、CMU Sphinx、DeepSpeech 都是比较流行的本地语音识别库。\n*   **训练模型：**  使用训练数据训练语音识别模型。\n*   **识别文本：** 使用训练好的模型识别语音。\n\n**3. 使用Web Speech API：**\n\n*   **浏览器支持：** 适用于在浏览器中运行的Web应用。\n*   **直接识别：**  Web Speech API可以直接在浏览器中识别语音，并将识别结果返回给你的JavaScript代码。\n\n**建议：**\n\n*   **对于简单的应用场景：** Web Speech API是一个不错的选择，它可以让你快速实现语音识别功能，而无需安装任何额外的软件。\n*   **对于需要更高准确率的应用场景：** 云端语音识别API通常提供更高的准确率，并且可以支持更多的语言和口音。\n*   **对于需要离线使用或保护隐私的应用场景：** 本地语音识别库是更好的选择，它可以让你在没有网络连接的情况下进行语音识别，并且可以保护你的数据隐私。\n\n**在选择方案时，还需要考虑以下因素：**\n\n*   **准确率：** 语音识别的准确率对用户体验至关重要。\n*   **延迟：** 语音识别的延迟会影响用户体验。\n*   **成本：** 云端语音识别API通常需要付费。\n*   **易用性：**  不同的语音识别方案的易用性不同。\n\n希望这些信息对你有所帮助。如果你需要更详细的指导，可以告诉我你使用的具体技术栈和应用场景，我会尽力为你提供更具体的建议。"
    for ch in text:
        result = segmenter.push(ch)
        if(result != []):
            print(result)

    # text = "第一句。\n\n第二句。\n\n\n第三句。"
    # result = StnSplit().split(text)
    # print(result)

