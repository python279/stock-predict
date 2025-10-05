"""
配置加载模块
"""

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"成功加载配置文件: {self.config_path}")
            return self.config
        
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置键路径，如 "email.sender_email"
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def validate_config(self) -> bool:
        """
        验证必要的配置项是否存在
        
        Returns:
            配置是否有效
        """
        required_keys = [
            'email.sender_email',
            'email.sender_password',
            'email.recipient_email',
            'llm.api_key',
            'llm.model'
        ]
        
        missing_keys = []
        for key in required_keys:
            value = self.get(key)
            if not value or str(value).startswith('your-'):
                missing_keys.append(key)
        
        if missing_keys:
            logger.warning(f"以下配置项未正确设置: {', '.join(missing_keys)}")
            return False
        
        logger.info("配置验证通过")
        return True

