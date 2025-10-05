"""
数据存储模块
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """数据存储管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据存储管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        storage_config = config.get('storage', {})
        
        self.save_raw_news = storage_config.get('save_raw_news', True)
        self.news_cache_dir = storage_config.get('news_cache_dir', 'data/news_cache')
        self.reports_dir = storage_config.get('reports_dir', 'data/reports')
        self.max_cache_days = storage_config.get('max_cache_days', 7)
        
        # 创建目录
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        for directory in [self.news_cache_dir, self.reports_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"创建目录: {directory}")
    
    def save_news_cache(self, articles: List[Any]) -> str:
        """
        保存新闻缓存
        
        Args:
            articles: 文章列表
        
        Returns:
            保存的文件路径
        """
        if not self.save_raw_news or not articles:
            return ""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"news_{timestamp}.json"
            filepath = os.path.join(self.news_cache_dir, filename)
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'count': len(articles),
                'articles': [article.to_dict() for article in articles]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存新闻缓存: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"保存新闻缓存失败: {e}")
            return ""
    
    def save_analysis_report(self, analysis_result: Dict[str, Any]) -> str:
        """
        保存分析报告
        
        Args:
            analysis_result: 分析结果
        
        Returns:
            保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存 JSON 格式
            json_filename = f"report_{timestamp}.json"
            json_filepath = os.path.join(self.reports_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存分析报告 (JSON): {json_filepath}")
            
            # 保存纯文本格式
            text_filename = f"report_{timestamp}.txt"
            text_filepath = os.path.join(self.reports_dir, text_filename)
            
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(f"全球新闻分析报告\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(f"分析时间: {analysis_result.get('analysis_time', '')}\n")
                f.write(f"文章数量: {analysis_result.get('articles_count', 0)}\n")
                f.write(f"覆盖地区: {', '.join(analysis_result.get('regions_covered', []))}\n\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(analysis_result.get('analysis', ''))
                
                # 添加参考资料部分
                all_articles = analysis_result.get('all_articles', [])
                if all_articles:
                    f.write("\n\n")
                    f.write(f"{'=' * 60}\n\n")
                    f.write("### 参考资料\n\n")
                    f.write(f"本报告基于以下 {len(all_articles)} 篇新闻进行分析：\n\n")
                    
                    # 按区域分组显示
                    regions = {}
                    for article in all_articles:
                        region = article.get('region', 'unknown')
                        if region not in regions:
                            regions[region] = []
                        regions[region].append(article)
                    
                    # 区域名称映射
                    region_names = {
                        'americas': '美洲地区',
                        'europe': '欧洲地区',
                        'asia': '亚洲地区',
                        'russia': '俄罗斯及独联体',
                        'global': '全球综合',
                        'unknown': '其他'
                    }
                    
                    for region, articles in sorted(regions.items()):
                        region_name = region_names.get(region, region)
                        f.write(f"#### {region_name}\n\n")
                        
                        for i, article in enumerate(articles, 1):
                            title = article.get('title', '无标题')
                            url = article.get('url', '')
                            source = article.get('source', '未知来源')
                            published_at = article.get('published_at', '')
                            
                            f.write(f"{i}. **{title}**\n")
                            f.write(f"   来源: {source}\n")
                            if published_at:
                                f.write(f"   时间: {published_at}\n")
                            f.write(f"   链接: {url}\n\n")
            
            logger.info(f"保存分析报告 (TXT): {text_filepath}")
            
            return json_filepath
        
        except Exception as e:
            logger.error(f"保存分析报告失败: {e}")
            return ""
    
    def clean_old_cache(self):
        """清理过期的缓存文件"""
        try:
            if not os.path.exists(self.news_cache_dir):
                return
            
            cutoff_date = datetime.now() - timedelta(days=self.max_cache_days)
            deleted_count = 0
            
            for filename in os.listdir(self.news_cache_dir):
                filepath = os.path.join(self.news_cache_dir, filename)
                
                if not os.path.isfile(filepath):
                    continue
                
                # 检查文件修改时间
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_mtime < cutoff_date:
                    os.remove(filepath)
                    deleted_count += 1
                    logger.debug(f"删除过期缓存: {filename}")
            
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个过期缓存文件")
        
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    def get_latest_report(self) -> Optional[str]:
        """
        获取最新的报告文件路径
        
        Returns:
            最新报告文件路径，如果没有则返回 None
        """
        try:
            if not os.path.exists(self.reports_dir):
                return None
            
            json_files = [
                f for f in os.listdir(self.reports_dir)
                if f.startswith('report_') and f.endswith('.json')
            ]
            
            if not json_files:
                return None
            
            # 按修改时间排序
            json_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.reports_dir, f)),
                reverse=True
            )
            
            return os.path.join(self.reports_dir, json_files[0])
        
        except Exception as e:
            logger.error(f"获取最新报告失败: {e}")
            return None

