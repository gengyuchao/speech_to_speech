import os
import json
import ollama
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from sentence_segmenter import SentenceSegmenter
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

segmenter = SentenceSegmenter()

# 创建线程池执行器用于处理同步IO
executor = ThreadPoolExecutor(max_workers=1)

default_model = "deepseek-r1" 
default_model = "gemma3:27b" 

AI_call_func_promot = """
辅助功能说明：

### 可用功能清单
1. 文件操作  
   - `CREATE_FILE`：创建文件（支持参数 `metadata.overwrite`）  
   - `DELETE_FILE`：删除文件  
   - `READ_FILE`：读取文件内容（支持 `metadata.encoding`）  
   - `WRITE_FILE`：写入内容（支持 `metadata.overwrite` 模式）  
   - `UPDATE_FILE`：修改指定行（需 `metadata.line_number`）  
2. 目录管理  
   - `CREATE_DIR`：创建目录（支持 `metadata.recursive`）  
   - `GET_DIR`：列出目录内容  
3. 系统命令  
   - `EXECUTE_COMMAND`：执行系统命令（`target` 为命令名，`metadata.command_args` 为参数列表）  

### 调用方法
- 默认用户：`user="AI"`
- 必填字段：`operation`（操作类型）、`target`（路径）。  
- 参数传递：通过 `metadata` 字段传递扩展参数（如 `overwrite`, `encoding`, `line_number` 等）。  
- 内容写入：需在 `content` 字段填写文本内容。  

### 示例指令
```json
# 写入文件
{
  "operation": "WRITE_FILE",
  "target": "/tmp/example.txt",
  "content": "Hello World",
  "metadata": {"overwrite": true}
}

# 执行命令
{
  "operation": "EXECUTE_COMMAND",
  "target": "echo",
  "metadata": {"command_args": ["Hello", "AI"]}
}

### 注意事项
- 路径需合法且权限充足（默认无权限，需预先配置）。  
- 文件操作会自动创建父目录（除 `CREATE_FILE` 外）。  
- 输出结果超过 100 字符会被截断。  
- 权限不足或路径错误会记录失败日志。  
"""

class ChatHistoryManager:
    def __init__(self, model: str = "deepseek-r1", max_history: int = 20, compress_interval: int = 5):
        """
        Args:
            model: 用于生成摘要的模型名称
            max_history: 历史记录最大长度
            compress_interval: 每隔多少条进行一次压缩
        """
        self.model = model
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.compress_interval = compress_interval
        self.total_turns = 0  # 总对话轮数

    async def add_message(self, role: str, content: str):
        """添加单条消息到历史"""
        self.history.append({"role": role, "content": content})
        self.total_turns += 1

    async def summarize_earliest(self, m: int = 5) -> Optional[str]:
        """用模型总结最早 m 条历史记录"""
        if len(self.history) < m:
            return None

        to_summarize = self.history[:m]
        summary_prompt = (
            "请对以下对话历史进行简洁的总结，保留关键信息和上下文关系。\n\n"
            + "\n".join([f"{msg['role']}: {msg['content']}" for msg in to_summarize])
        )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                partial(ollama.chat, model=self.model, messages=[{"role": "user", "content": summary_prompt}])
            )
            summary = response["message"]["content"]
            # 替换掉前 m 条为总结
            self.history = [{"role": "system", "content": f"[历史摘要] {summary}"}] + self.history[m:]
            return summary
        except Exception as e:
            print(f"历史总结失败: {e}")
            return None

    async def maybe_compress_history(self):
        """判断是否需要压缩历史"""
        if self.total_turns >= self.max_history and (self.total_turns % self.compress_interval == 0):
            await self.summarize_earliest(m=self.compress_interval)

    async def get_messages_for_model(self) -> List[Dict[str, str]]:
        """获取当前历史作为模型输入"""
        return self.history.copy()


    def save_to_file(self, filepath: str = "history.json"):
        """将当前历史记录保存到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "total_turns": self.total_turns,
                "history": self.history
            }, f, ensure_ascii=False, indent=2)
        print(f"[+] 已保存历史到 {filepath}")

    def load_from_file(self, filepath: str = "history.json"):
        """从文件加载历史记录"""
        if not os.path.exists(filepath):
            print(f"[!] 文件 {filepath} 不存在，跳过加载")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.total_turns = data.get("total_turns", 0)
                self.history = data.get("history", [])
            print(f"[+] 已从 {filepath} 加载历史记录")
        except Exception as e:
            print(f"[!] 加载历史出错: {e}")

# 全局历史管理器实例
history_manager = ChatHistoryManager(model=default_model, max_history=20, compress_interval=5)
history_manager.load_from_file()  # 启动时尝试加载

# 获取当前本地时间
local_time = None
formatted_local = None

async def stream_chat(prompt, model=default_model, speaker_id="unknown"):
    """
    异步流式处理函数，分离思考过程和回复内容
    
    Args:
        prompt: 用户输入文本
        model: 使用的模型名称
        speaker_id: 说话人标识
        
    Yields:
        dict: 包含类型和内容的事件对象，格式为:
            {'type': 'thinking', 'content': '...'} 或
            {'type': 'response', 'content': '...'}
    """
    global local_time, formatted_local
    # 获取当前历史
    messages = await history_manager.get_messages_for_model()

    last_time = local_time
    print(f"1 last_time : {last_time}")
    local_time = datetime.now()
    print(f"2 last_time : {last_time}")
    formatted_local = local_time.strftime("%Y年%m月%d日%H时%M分%S秒")

    # 添加系统提示
    system_prompt = [
        {"role": "system", "content": f"你是超强的人工智能助手钟离，你正在和 {speaker_id} 对话。"},
        {"role": "system", "content": f"使用自然对话的说话方式，只输出中文文字和标点，不输出阿拉伯数字和特殊符号。"},
        {"role": "system", "content": f"你可以使用 json 输出命令:{AI_call_func_promot}"},
    ]

    # 判断是否需要添加时间提示
    if last_time is None or (local_time - last_time) > timedelta(minutes=10):
        environment_prompt = [{
            "role": "system",
            "content": f"当前时间是 {formatted_local}，请根据时间进行适当的回应。"
        }]
        last_time = local_time  # 更新 last_time
    else :
        environment_prompt = []
        print(f"last time is {last_time}")
        print(f"formatted_local is {formatted_local}")
        
    # 构造最终 messages
    full_messages = system_prompt + messages + environment_prompt + [{"role": "user", "content": prompt}]

    print(full_messages)

    loop = asyncio.get_event_loop()

    if model == "deepseek-r1":
        think_enable = True
    else :
        think_enable = False 


    # 在线程池中执行同步的流式请求
    response_stream = await loop.run_in_executor(
        executor,
        lambda: ollama.chat(
            model=model,
            messages=full_messages,
            think=think_enable,
            stream=True
        )
    )

    # 初始化缓冲区
    # 初始化缓冲区
    thinking_buffer = ""
    response_buffer = ""
    thinking_started = False
    thinking_ended = False

    try:
        # 处理流式响应
        for chunk in response_stream:
            if hasattr(chunk, 'message'):
                # 处理思考过程
                if chunk.message.thinking:
                    thinking_buffer += chunk.message.thinking
                    if not thinking_started:
                        yield {
                            'type': 'thinking',
                            'content': '[思考过程] ' + chunk.message.thinking
                        }
                        thinking_started = True
                    else:
                        yield {
                            'type': 'thinking',
                            'content': chunk.message.thinking
                        }
                elif thinking_started and not thinking_ended:
                    yield {
                        'type': 'thinking',
                        'content': '\n[思考过程结束]'
                    }
                    thinking_ended = True
                    thinking_started = False
                        
                # 处理回复内容
                if chunk.message.content:
                    response_buffer += chunk.message.content

                    sentences = segmenter.push(chunk.message.content)

                    for sentence in sentences:
                        yield {'type': 'response', 'content': sentence}

        # 输出剩余内容
        if segmenter.buffer != "":
            yield {'type': 'response', 'content': segmenter.buffer}
            segmenter.buffer = ""

 
        # 思考结束
        if thinking_started and not thinking_ended:
            yield {
                'type': 'thinking',
                'content': '\n[思考过程结束]'
            }
            thinking_ended = True

    except Exception as e:
        yield {'type': 'error', 'content': f"流式处理异常: {str(e)}"}
    finally:
        # 保存用户输入和模型输出到历史
        await history_manager.add_message("user", prompt)
        await history_manager.add_message("assistant", response_buffer.strip())
        await history_manager.maybe_compress_history()
        history_manager.save_to_file()  # 每次对话后自动保存

async def chat_handle(user_prompt,speaker_name, tts_handler):
    async for event in stream_chat(user_prompt, speaker_id=speaker_name):
        if event['type'] == 'thinking':
            print(event['content'], end='', flush=True)
        elif event['type'] == 'response':
            print(f"[回复内容] {event['content']}")
            if tts_handler != None:
                tts_handler(event['content'])
        elif event['type'] == 'error':
            print(f"[错误] {event['content']}")

# if __name__ == "__main__":
#     asyncio.run(main())
