import re
import json
from collections import defaultdict

class LLMJSONAnalyzer:
    def __init__(self):
        self.json_pattern = re.compile(
            r'```json\s*(\{.*?\})\s*```|(\{.*?\})',  # 匹配 ```json 或裸 JSON
            re.DOTALL | re.MULTILINE
        )
        
    def extract_jsons(self, text):
        """从文本中提取所有 JSON 内容"""
        matches = self.json_pattern.findall(text)
        raw_jsons = [m[0] or m[1] for m in matches if m[0] or m[1]]
        
        cleaned = []
        for j in raw_jsons:
            j = j.strip()
            j = j.replace('\uFEFF', '')  # 移除 BOM 头
            j = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', j)  # 移除控制字符
            # 修复多余的转义引号（如 \"operation\" → "operation"）
            j = re.sub(r'\\(")', r'\1', j)  # 将 \" 替换为 "（仅保留必要转义）
            cleaned.append(j)
        return cleaned
    
    def preprocess_json_string(self, json_str):
        """修复 JSON 中的非法换行符和转义"""
        # 修复未转义的换行符（将 \n 替换为 \\n）
        json_str = re.sub(r'(?<!\\)(\\n)', r'\n', json_str)  # 处理已错误转义的 \n
        return json_str

    def parse_json(self, json_str):
        """解析单个 JSON 字符串"""
        try:
            # 预处理非法换行符和多余引号
            json_str = self.preprocess_json_string(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 尝试修复常见错误（如末尾逗号、键名未加引号）
            if "Expecting value" in str(e):
                # 修复键名未加引号的问题
                json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
                # 修复字符串值未加引号的问题
                json_str = re.sub(r':\s*([^"\{\}\[\],]+?)\s*([,}])', r': "\1"\2', json_str)
                # 再次预处理
                json_str = self.preprocess_json_string(json_str)
                return json.loads(json_str)
            raise ValueError(f"JSON 解析失败: {str(e)}\n原始内容: {json_str[:200]}...")
        
    def analyze_structure(self, data, parent_key="root"):
        """递归分析 JSON 结构"""
        if isinstance(data, dict):
            structure = {}
            for key, value in data.items():
                structure[key] = self.analyze_structure(value, f"{parent_key}.{key}")
            return {"type": "object", "fields": structure}
        elif isinstance(data, list):
            if not data:
                return {"type": "array", "items": "unknown"}
            return {"type": "array", "items": self.analyze_structure(data[0], parent_key)}
        else:
            return {"type": type(data).__name__}
    
    def generate_structure_vars(self, json_data):
        """生成扁平化变量表示"""
        structure = self.analyze_structure(json_data)
        return self._flatten_structure(structure)
    
    def _flatten_structure(self, structure, prefix=""):
        """展平嵌套结构"""
        result = {}
        if structure["type"] == "object":
            for key, value in structure.get("fields", {}).items():
                sub_vars = self._flatten_structure(value, f"{prefix}.{key}" if prefix else key)
                result.update(sub_vars)
        elif structure["type"] == "array":
            result[prefix] = f"Array<{structure['items']['type']}>"
        else:
            result[prefix] = structure["type"]
        return result
    
    def process_text(self, input_text):
        """处理完整文本流程"""
        json_strings = self.extract_jsons(input_text)
        results = []
        
        for idx, json_str in enumerate(json_strings):
            try:
                data = self.parse_json(json_str)
                vars_dict = self.generate_structure_vars(data)
                results.append({
                    "json_index": idx + 1,
                    "structure": vars_dict,
                    "sample": json.dumps(data, ensure_ascii=False)
                })
            except Exception as e:
                print(f"[警告] 第 {idx + 1} 个 JSON 解析失败: {str(e)}")
                
        return results

def get_nested_value(d, keys):
    """从嵌套字典中安全获取字段值"""
    from functools import reduce
    try:
        return reduce(lambda x, y: x[y], keys, d)
    except (KeyError, TypeError, IndexError):
        return None
        
def main():
    import sys
    if len(sys.argv) < 2:
        print("用法: python llm_json_analyzer.py \"路径/文本 或 -t '直接文本'\"")
        return
    
    input_source = sys.argv[1]
    
    if input_source == "-t" and len(sys.argv) >= 3:
        input_text = sys.argv[2]
    else:
        try:
            with open(input_source, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except Exception as e:
            input_text = input_source  # 当作直接文本处理
    
    analyzer = LLMJSONAnalyzer()
    results = analyzer.process_text(input_text)

    print(results)
    
    for res in results:
        print(res["sample"])

        try:
            # 将 sample 转换为字典用于取值
            sample_dict = json.loads(res['sample'])
        except json.JSONDecodeError:
            print(f"[警告] 样本 JSON 解析失败：{res['sample']}")
            continue

        print(f"\n{'=' * 80}")
        print(f"【JSON 结构 #{res['json_index']}】")
        print(f"示例内容: {res['sample']}")
        print(f"{'-' * 80}")

        for field, field_type in res['structure'].items():
            # 获取字段值（支持嵌套）
            value = get_nested_value(sample_dict, field.split('.'))
            value_str = str(value) if value is not None else "null"

            # 输出格式
            print(f"• {field.ljust(30)} | 类型: {field_type:<10} | 值: {value_str[:100]}")

        print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()