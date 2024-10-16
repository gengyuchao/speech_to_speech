import openai
import asyncio
import re

import string


openai.api_key = 'xxxxxx'
openai.base_url = "http://localhost:1234/v1/"


def is_all_special_characters(s):
    # 定义特殊字符集合
    special_characters = set(string.punctuation)
    # 检查字符串中的每个字符是否都在特殊字符集合中
    return all(char in special_characters for char in s)


class OpenAIHandler:
    def __init__(self):
        self.chat_history = [
            {"role": "system", "content": "你是聪明的人工智能，你的话语简洁，字符使用简体中文，不加任何特殊字符。"},
            {"role": "system", "content": "我们在进行语音对语音的对话，你的任何输出都会被转成音频，所以不要添加特殊符号或者括号。"}
            ]

    def add_to_history(self, ai_response):
        self.chat_history.append({"role": "assistant", "content": ai_response})

        if len(self.chat_history) > 5:
            self.chat_history.pop(1)

    def get_openai_response(self, prompt, output_queue):
        """
        与OpenAI API交互，并将结果放入队列中。

        :param self: 
        :param prompt: 
        """
        self.chat_history.append({"role": "user", "content": prompt})

        ai_response = ""
        current_sentence = ""

        stream = openai.chat.completions.create(
            model="gpt-4",
            messages=self.chat_history,
            stream=True,
            temperature=0.7,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content or ""
                ai_response += delta
                current_sentence += delta
                print(delta, end="", flush=True)
                # 句子断句逻辑
                if len(current_sentence) > 4 and re.search(r'[.!?；：！？。]\s*|\n', current_sentence):

                    new_line_pos = current_sentence.find('\n')
                    if new_line_pos >= 0 :
                        sentence = current_sentence[:new_line_pos].strip().replace("*","")
                        # print(f"find :{new_line_pos}")
                        if sentence != "":
                            output_queue.put(sentence)
                        current_sentence = current_sentence[new_line_pos+1:]
                    else :

                        # 使用正则表达式查找标点符号，忽略数字后的点号
                        match = re.search(
                            r'(.*?)(?<!\d)[.!?；：！？。]\s*|\n', current_sentence.strip())
                        if match:
                            sentence = match.group(0).strip().replace("*","")
                            if sentence != "":
                                output_queue.put(sentence)
                            else:
                                current_sentence = "" # 可能有其他字符，如换行符，无法朗读
                                # print(f"sentence 是空的")
                                # print(match)
                            current_sentence = current_sentence[len(match.group(0)):]
               

        # 如果最后一句话没有结束符，可以在这里处理
        if current_sentence:
            sentence = current_sentence.strip()
            if sentence != "":
                output_queue.put(sentence)
            print("")

        # 添加完整的历史记录
        self.add_to_history(ai_response)
