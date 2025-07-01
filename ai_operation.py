import os
import json
import logging
import subprocess
from datetime import datetime

# 配置日志系统
logging.basicConfig(
    filename='file_operations.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 权限配置：每个用户允许的操作
USER_PERMISSIONS = {
    # "admin": ["CREATE_FILE", "DELETE_FILE", "READ_FILE", "WRITE_FILE", "UPDATE_FILE", "CREATE_DIR", "GET_DIR", "EXECUTE_COMMAND"],
    "alice": ["READ_FILE", "WRITE_FILE", "GET_DIR"],
    "AI": ["READ_FILE", "GET_DIR"],
}

def log_action(user, operation, target, result, success=True):
    status = "SUCCESS" if success else "FAILURE"
    logging.info(f"{status} | User: {user} | Operation: {operation} | Target: {target} | Result: {result}")

def execute_instruction(instruction):
    user = instruction.get("user", "anonymous")
    op = instruction.get("operation")
    target = instruction.get("target")
    content = instruction.get("content", "")
    metadata = instruction.get("metadata", {})

    # 权限检查
    if op not in USER_PERMISSIONS.get(user, []):
        msg = f"Permission denied for user '{user}' on operation '{op}'."
        log_action(user, op, target, msg, success=False)
        return msg

    try:
        if op == "CREATE_FILE":
            if os.path.exists(target) and not metadata.get("overwrite", False):
                result = f"File {target} already exists."
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                open(target, 'w').close()
                result = f"File {target} created."

        elif op == "DELETE_FILE":
            os.remove(target)
            result = f"File {target} deleted."

        elif op == "READ_FILE":
            with open(target, 'r', encoding=metadata.get("encoding", "utf-8")) as f:
                result = f.read()

        elif op == "WRITE_FILE":
            mode = 'w' if metadata.get("overwrite", False) else 'a'
            with open(target, mode, encoding=metadata.get("encoding", "utf-8")) as f:
                f.write(content)
            result = f"Wrote to {target}."

        elif op == "UPDATE_FILE":
            line_no = metadata.get("line_number")
            with open(target, 'r+', encoding="utf-8") as f:
                lines = f.readlines()
                if 0 <= line_no - 1 < len(lines):
                    lines[line_no - 1] = content + '\n'
                    f.seek(0)
                    f.writelines(lines)
                    result = f"Updated line {line_no} in {target}."
                else:
                    result = f"Line number {line_no} out of range."

        elif op == "CREATE_DIR":
            os.makedirs(target, exist_ok=metadata.get("recursive", True))
            result = f"Directory {target} created."

        elif op == "GET_DIR":
            result = os.listdir(target)

        elif op == "EXECUTE_COMMAND":
            cmd = [target] + metadata.get("command_args", [])
            completed = subprocess.run(cmd, capture_output=True, text=True)
            result = completed.stdout or completed.stderr

        else:
            result = "Unknown operation."

        log_action(user, op, target, str(result)[:100], success=True)  # truncate long output
        return result

    except Exception as e:
        log_action(user, op, target, str(e), success=False)
        return f"Error: {e}"


test_json = {
  "user": "bob",
  "operation": "READ_FILE",
  "target": "../usefull_cmd.md",
  "metadata": {
    "encoding": "utf-8"
  }
}


result = execute_instruction(test_json)
print(result)
