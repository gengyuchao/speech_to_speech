import logging
import os
from pathlib import Path

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    # 创建日志目录
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# 创建全局日志记录器
system_logger = setup_logger("System", "./logs/system.log", logging.INFO)