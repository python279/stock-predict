"""
调度器模块
支持定时执行新闻分析任务
"""

import schedule
import time
import logging
from datetime import datetime
import pytz
from typing import Dict, Any
import sys

from main import NewsAnalyzerApp

logger = logging.getLogger(__name__)


class NewsAnalyzerScheduler:
    """新闻分析调度器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化调度器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.app = NewsAnalyzerApp(config_path)
        
        # 获取调度配置
        scheduler_config = self.app.config.get('scheduler', {})
        self.enabled = scheduler_config.get('enabled', True)
        self.run_time = scheduler_config.get('run_time', '08:00')
        self.timezone_str = scheduler_config.get('timezone', 'Asia/Shanghai')
        
        try:
            self.timezone = pytz.timezone(self.timezone_str)
        except:
            logger.warning(f"无效的时区: {self.timezone_str}，使用 UTC")
            self.timezone = pytz.UTC
    
    def run_task(self):
        """执行任务"""
        try:
            logger.info("=" * 60)
            logger.info(f"定时任务触发: {datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            logger.info("=" * 60)
            
            self.app.run()
        
        except Exception as e:
            logger.error(f"定时任务执行失败: {e}")
    
    def start(self):
        """启动调度器"""
        if not self.enabled:
            logger.warning("调度器未启用，将立即执行一次任务后退出")
            self.run_task()
            return
        
        logger.info("=" * 60)
        logger.info("新闻分析调度器启动")
        logger.info(f"运行时间: 每天 {self.run_time}")
        logger.info(f"时区: {self.timezone_str}")
        logger.info("=" * 60)
        
        # 设置定时任务
        schedule.every().day.at(self.run_time).do(self.run_task)
        
        # 显示下次运行时间
        next_run = schedule.next_run()
        if next_run:
            logger.info(f"下次运行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 主循环
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        
        except KeyboardInterrupt:
            logger.info("调度器已停止")
        
        except Exception as e:
            logger.error(f"调度器异常: {e}")
            raise


def main():
    """主函数"""
    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        
        scheduler = NewsAnalyzerScheduler(config_path)
        scheduler.start()
    
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"调度器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

