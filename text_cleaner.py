# text_cleaner.py
import re

class TextCleaner:
    def __init__(self):
        # 可配置规则
        self.rules = [
            self._remove_markdown_symbols,     # 移除 Markdown 符号
            self._remove_unsupported_symbols,  # 移除不支持的字符
            self._normalize_punctuation,       # 统一中英文标点
            self._simplify_brackets,           # 简化括号内容（可选）
            self._replace_emojis,              # 替换表情为文字描述（可选）
            self._normalize_numbers,           # 数字格式标准化
            self._remove_whitespace,           # 移除所有空白字符（可按需调整）
        ]

    def clean(self, text):
        for rule in self.rules:
            text = rule(text)
        return text.strip()

    def _remove_markdown_symbols(self, text):
        # 删除 Markdown 中常见的符号
        return re.sub(r'[\*\_\`\#\-\>\!\$\[\]]', '', text)

    def _remove_unsupported_symbols(self, text):
        # 删除其他可能影响 TTS 的字符
        return re.sub(r'[~《》<>“”【】※…]', '', text)

    def _normalize_punctuation(self, text):
        # 统一中英文标点为中文标点
        punctuation_map = {
            ',': '，',
            ';': '；',
            '?': '？',
            '!': '！',
            '.': '。',
            '(': '（',
            ')': '）',
            '[': '【',
            ']': '】',
            # 部分中文符号也会导致生成音频异常，在此替换
            "'": " ",
            '：': '，',
            ':': '，',
            '“': '',
            '”': '',
            '"': '',
            "—": ' ',

        }
        return ''.join(punctuation_map.get(c, c) for c in text)

    def _simplify_brackets(self, text):
        # 删除括号及其中内容（可改为保留）
        return re.sub(r'（[^）]*）', '', text)

    def _replace_emojis(self, text):
        # 简单替换部分 emoji 为可读文本（示例）
        emoji_replacement = {
            '❤️': '爱心',
            '↑': '向上箭头',
            '↓': '向下箭头',
            '✅': '对勾',
            '⚠️': '警告标志',
            '➡️': '向右箭头'
        }
        for emoji, replacement in emoji_replacement.items():
            text = text.replace(emoji, replacement)
        return text

    def _normalize_numbers(self, text):
        # 将百分比等转为中文表达
        text = text.replace('%', '百分之')
        return text
        
    def _remove_whitespace(self, text):
        # 移除所有空白字符：空格、Tab、换行等
        return re.sub(r'\s+', '', text)

if __name__ == "__main__":
    cleaner = TextCleaner()
    
    raw_text = """
    *这是一个测试句子*，包含一些特殊符号（如 ❤️ 和 ↑），
    还有#标签和[链接](http://example.com)，以及 100% 成功率！
    """

    cleaned = cleaner.clean(raw_text)
    print(cleaned)
    