import logging
import os
from pathlib import Path
from config_manager import config_manager

# 从配置文件导入参数
# 使用统一配置管理器

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
system_logger = setup_logger("System", config_manager.get('logging.file'), getattr(logging, config_manager.get('logging.level')))