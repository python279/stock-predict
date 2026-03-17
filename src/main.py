"""
主程序模块
协调各个模块完成新闻抓取、分析和发送
"""

import logging
import sys
import traceback
from datetime import datetime
from typing import Optional

from config_loader import ConfigLoader
from logger_config import setup_logger
from news_fetcher import NewsFetcher
from llm_analyzer import LLMAnalyzer
from email_sender import EmailSender
from data_storage import DataStorage

logger = logging.getLogger(__name__)


class NewsAnalyzerApp:
    """新闻分析应用主类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化应用
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # 设置日志
        setup_logger(self.config)
        logger.info("=" * 60)
        logger.info("全球新闻分析系统启动")
        logger.info("=" * 60)
        
        # 验证配置
        if not self.config_loader.validate_config():
            logger.warning("配置验证未通过，某些功能可能无法正常工作")
        
        # 初始化各个模块
        try:
            self.fetcher = NewsFetcher(self.config)
            self.analyzer = LLMAnalyzer(self.config)
            self.email_sender = EmailSender(self.config)
            self.storage = DataStorage(self.config)
            
            logger.info("所有模块初始化成功")
        except Exception as e:
            logger.error(f"模块初始化失败: {e}")
            raise
    
    def run(self) -> bool:
        """
        运行完整的新闻分析流程
        
        Returns:
            是否执行成功
        """
        try:
            start_time = datetime.now()
            logger.info(f"开始执行新闻分析任务: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 1. 抓取新闻
            logger.info("步骤 1/5: 抓取新闻...")
            articles = self.fetcher.fetch_all_news()
            
            if not articles:
                logger.warning("未抓取到任何新闻文章")
                self._send_error_notification("未能抓取到新闻")
                return False
            
            logger.info(f"成功抓取 {len(articles)} 篇文章")
            
            # 2. 保存新闻缓存
            logger.info("步骤 2/5: 保存新闻缓存...")
            self.storage.save_news_cache(articles)

            # 2.5 加载过去5天历史新闻（用于黑天鹅对比分析）
            logger.info("步骤 2.5/5: 加载历史新闻缓存（过去5天）...")
            historical_news = self.storage.load_historical_news_cache(days=5)
            if historical_news:
                logger.info(
                    f"已加载 {len(historical_news)} 天历史数据: "
                    + ", ".join(sorted(historical_news.keys()))
                )
            else:
                logger.info("无历史新闻缓存，仅使用今日数据")

            # 3. 分析新闻
            logger.info("步骤 3/5: 使用大模型分析新闻...")
            analysis_result = self.analyzer.analyze_news(articles, historical_news=historical_news)
            
            if not analysis_result or not analysis_result.get('analysis'):
                logger.error("新闻分析失败")
                self._send_error_notification("新闻分析失败")
                return False
            
            logger.info("新闻分析完成")
            
            # 4. 保存分析报告
            logger.info("步骤 4/5: 保存分析报告...")
            report_path = self.storage.save_analysis_report(analysis_result)
            logger.info(f"报告已保存: {report_path}")
            
            # 5. 发送邮件
            logger.info("步骤 5/5: 发送分析报告邮件...")
            email_success = self.email_sender.send_analysis_report(analysis_result)
            
            if not email_success:
                logger.error("邮件发送失败")
                return False
            
            # 6. 清理旧缓存
            logger.info("清理过期缓存...")
            self.storage.clean_old_cache()
            
            # 完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info(f"任务执行成功！耗时: {duration:.2f} 秒")
            logger.info("=" * 60)
            
            return True
        
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            logger.error(traceback.format_exc())
            self._send_error_notification(f"任务执行异常: {str(e)}")
            return False
    
    def _send_error_notification(self, error_message: str):
        """
        发送错误通知邮件
        
        Args:
            error_message: 错误信息
        """
        try:
            subject = "⚠️ 全球新闻分析系统 - 执行失败"
            html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; padding: 20px;">
    <h2 style="color: #dc3545;">任务执行失败</h2>
    <p><strong>时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>错误信息:</strong></p>
    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px;">
        {error_message}
    </div>
    <p style="margin-top: 20px; color: #666;">请检查日志文件获取详细信息</p>
</body>
</html>
"""
            plain_body = f"任务执行失败\n\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n错误: {error_message}"
            
            self.email_sender.send_email(
                subject=subject,
                html_body=html_body,
                plain_body=plain_body
            )
        except Exception as e:
            logger.error(f"发送错误通知失败: {e}")


def main():
    """主函数"""
    try:
        # 检查命令行参数
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        
        # 创建并运行应用
        app = NewsAnalyzerApp(config_path)
        success = app.run()
        
        # 根据执行结果设置退出码
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"程序异常退出: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

