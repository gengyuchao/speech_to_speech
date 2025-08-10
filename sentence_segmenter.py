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

        # # 处理普通句子分割
        # sentence_results = self._process_sentences(new_text)
        # results.extend(sentence_results)

        return results

    def _process_speech_stream(self, new_text):
        """流式解析说话人结构"""
        results = []
        self.speech_buffer += new_text
        
        # print(f"DEBUG: new_text='{new_text}'")
        # print(f"DEBUG: state='{self.state}', speech_buffer='{self.speech_buffer}'")

        while True:
            # print(f"DEBUG: 循环开始 - state='{self.state}', speech_buffer='{self.speech_buffer}'")
            
            if self.state == 'idle':
                # 查找说话人开始标记
                start_idx = self.speech_buffer.find(self.SPEAKER_START)
                # print(f"DEBUG: idle状态查找开始标记'{self.SPEAKER_START}'，位置={start_idx}")
                
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
                            print(f"DEBUG: 返回前缀文本: '{prefix_text}'")
                    
                    self.speech_buffer = self.speech_buffer[start_idx + len(self.SPEAKER_START):]
                    self.state = 'speaker'
                    self.speaker_buffer = ""
                    self.speaker_buffer += self.speech_buffer
                    # print(f"DEBUG: 找到开始标记，切换到speaker状态, 新buffer='{self.speech_buffer}'")
                else:
                    # 没找到开始标记，保留buffer等待更多数据
                    # print(f"DEBUG: 未找到开始标记，保持idle状态，buffer='{self.speech_buffer}'")
                    break

            elif self.state == 'speaker':
                # 查找说话人结束标记
                end_idx = self.speech_buffer.find(self.SPEAKER_END)
                # print(f"DEBUG: speaker状态查找结束标记'{self.SPEAKER_END}'，位置={end_idx}")
                # print(f"DEBUG: self.speech_buffer='{self.speech_buffer}' end_idx={end_idx}")
                
                if end_idx != -1:
                    # 找到了说话人结束标记
                    speaker_name = self.speech_buffer[:end_idx]
                    self.speaker_buffer = speaker_name
                    self.speech_buffer = self.speech_buffer[end_idx + len(self.SPEAKER_END):]
                    self.state = 'content'
                    self.content_buffer = ""
                    self.content_buffer += self.speech_buffer
                    # print(f"DEBUG: 找到结束标记，speaker='{speaker_name}', 新buffer='{self.speech_buffer}'")

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
                    # print(f"DEBUG: 未找到结束标记，累积到speaker_buffer='{self.speaker_buffer}'")
                    break

            elif self.state == 'content':
                # 查找说话结束标记
                end_idx = self.speech_buffer.find(self.SPEECH_END)
                # print(f"DEBUG: self.speech_buffer'{self.speech_buffer}'")
                # print(f"DEBUG: content状态查找结束标记'{self.SPEECH_END}'，位置={end_idx}")
                
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

                    # flush_results = self.flush(msg_type='speech',
                    # speaker=self.speaker_buffer.strip())

                    # 添加分句结果到最终返回列表
                    results.extend(sentence_results)
                    # results.extend(flush_results)

                    # 处理剩余部分
                    self.speech_buffer = self.speech_buffer[end_idx + len(self.SPEECH_END):]
                    self.state = 'idle'
                    self.speaker_buffer = ""
                    self.content_buffer = ""
                    # print(f"DEBUG: 完成结构处理，剩余buffer='{self.speech_buffer}'")

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

                    # print(f"DEBUG: 未找到结束标记，累积到content_buffer='{self.content_buffer}'")
                    break

        # print(f"DEBUG: 最终结果={results}")
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
        # print(f"self.buffer:{self.buffer}:: end_idx:{end_idx}")
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
    text = f"啊，是这样啊！我明白了。声音识别，也就是语音转文本（Speech-to-Text，STT）！\n\n{segmenter.SPEAKER_START}钟离{segmenter.SPEAKER_END}看来你希望在远程使用之前，先让程序能够识别你的声音指令，对吗？\n\n那确实是更基础的问题，优先解决语音识别是正确的。\n\n那么，解决语音识别的问题，也有几种方案：\n\n**1. 使用云端语音识别API：**\n\n*   **选择API：** 谷歌云语音识别 (Google Cloud Speech-to-Text)、亚马逊转录 (Amazon Transcribe)、微软 Azure 语音服务 (Microsoft Azure Speech Services) 都是不错的选择。\n*   **上传音频：** 将录制到的音频上传到云端服务器。\n*   **识别文本：** 使用云端API将音频转换为文本。\n*   **传输文本：** 将识别到的文本传输到本地或远程服务器。\n\n**2. 使用本地语音识别库：**\n\n*   **选择库：** Kaldi、CMU Sphinx、DeepSpeech 都是比较流行的本地语音识别库。\n*   **训练模型：**  使用训练数据训练语音识别模型。\n*   **识别文本：** 使用训练好的模型识别语音。\n\n**3. 使用Web Speech API：**\n\n*   **浏览器支持：** 适用于在浏览器中运行的Web应用。\n*   **直接识别：**  Web Speech API可以直接在浏览器中识别语音，并将识别结果返回给你的JavaScript代码。\n\n**建议：**\n\n*   **对于简单的应用场景：** Web Speech API是一个不错的选择，它可以让你快速实现语音识别功能，而无需安装任何额外的软件。\n*   **对于需要更高准确率的应用场景：** 云端语音识别API通常提供更高的准确率，并且可以支持更多的语言和口音。\n*   **对于需要离线使用或保护隐私的应用场景：** 本地语音识别库是更好的选择，它可以让你在没有网络连接的情况下进行语音识别，并且可以保护你的数据隐私。{segmenter.SPEECH_END}\n\n**在选择方案时，还需要考虑以下因素：**\n\n*   **准确率：** 语音识别的准确率对用户体验至关重要。\n*   **延迟：** 语音识别的延迟会影响用户体验。\n*   **成本：** 云端语音识别API通常需要付费。\n*   **易用性：**  不同的语音识别方案的易用性不同。\n\n希望这些信息对你有所帮助。如果你需要更详细的指导，可以告诉我你使用的具体技术栈和应用场景，我会尽力为你提供更具体的建议。"
    # text = f"啊，是这样啊！我明白了。声音识别，也就是语音转文本（Speech-to-Text，STT）！\n\n{segmenter.SPEAKER_START}钟离{segmenter.SPEAKER_END}看来你希望在远程使用之前，先让程序能够识别你的声音指令，对吗？{segmenter.SPEECH_END}\n\n那确实是更基础的问题，优先解决语音识别是正确的。"
    # text = f"[[/speaker_start]钟离[/speaker_end]]如此良辰美景，玉超想知道我能做什么吗？{segmenter.SPEECH_END}"
    # for ch in text:
    #     result = segmenter.push(ch)
    #     if result:
    #         print(result)

    # final_result = segmenter.flush()
    # if final_result:
    #     print(final_result)


    # text = f"[[/speaker_start]钟离[/speaker_end]]如此良辰美景，玉超想知道我能做什么吗？{segmenter.SPEECH_END}"
    
    # text = f"[[/speaker_start]派蒙[/speaker_end]]啊，现在是傍晚了吗？时间过得真快。介绍我的特点吗？嗯……我拥有悠久的历史，曾经是统治璃月港的岩王帝君。现在，我更喜欢以一个顾问的身份，享受闲适的生活。我喜欢品尝美食，研究历史，也乐于结交朋友。哦，对了，我还非常擅长签订契约，如果你有什么需要，随时可以告诉我。不过，最重要的是，我拥有着一颗热爱璃月、守护璃月的心。你觉得呢，玉超？{segmenter.SPEECH_END}[[/speaker_start]钟离[/speaker_end]]如此良辰美景，玉超想知道我能做什么吗？{segmenter.SPEECH_END}"
    text += "[[/speaker_start]钟离[/speaker_end]]如此，修复错误是好事。\n[/say_end]\n[[/speaker_start]"
    text += "[[/speaker_start]温迪[/speaker_end]]哦？看来你终于意识到需要我的声音了。真是令人惊喜啊。\n[/say_end]"
    # 两个字两个字地切分
    chunks = [text[i:i+3] for i in range(0, len(text), 3)]
    for chunk in chunks:
        result = segmenter.push(chunk)
        if result:
            for item in result:
                print(item)
                