"""
数据存储模块
"""

import json
import os
import re
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
        self.mobile_reports_dir = storage_config.get(
            'mobile_reports_dir', 'data/mobile_reports'
        )
        self.market_cache_dir = storage_config.get('market_cache_dir', 'data/market_cache')
        self.max_cache_days = storage_config.get('max_cache_days', 7)
        
        # 创建目录
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        for directory in [
            self.news_cache_dir,
            self.reports_dir,
            self.mobile_reports_dir,
            self.market_cache_dir,
        ]:
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

    def save_market_cache(
        self,
        market_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
    ) -> str:
        """保存本次市场、政策/新闻情绪快照，供审计和后续扩展使用。"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.market_cache_dir, f"market_{timestamp}.json")
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(
                    {
                        'timestamp': datetime.now().isoformat(),
                        'market_data': market_data,
                        'sentiment_data': sentiment_data,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"保存市场缓存: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存市场缓存失败: {e}")
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
            
            # 保存 Markdown 格式
            markdown_filename = f"report_{timestamp}.md"
            markdown_filepath = os.path.join(self.reports_dir, markdown_filename)
            
            with open(markdown_filepath, 'w', encoding='utf-8') as f:
                f.write("# 全球新闻分析报告\n\n")
                f.write(f"分析时间: {analysis_result.get('analysis_time', '')}\n")
                f.write(f"文章数量: {analysis_result.get('articles_count', 0)}\n")
                f.write(f"大宗商品数量: {analysis_result.get('commodities_count', 0)}\n")
                f.write(f"市场行情数量: {analysis_result.get('market_items_count', 0)}\n")
                f.write(f"覆盖地区: {', '.join(analysis_result.get('regions_covered', []))}\n\n")
                f.write("---\n\n")
                f.write(analysis_result.get('analysis', ''))
                
                # 添加参考资料部分
                all_articles = analysis_result.get('all_articles', [])
                if all_articles:
                    f.write("\n\n")
                    f.write("---\n\n")
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
                            f.write(f"   - 来源: {source}\n")
                            if published_at:
                                f.write(f"   - 时间: {published_at}\n")
                            if url:
                                f.write(f"   - 链接: <{url}>\n")
                            f.write("\n")
            
            logger.info(f"保存分析报告 (Markdown): {markdown_filepath}")
            
            mobile_filepath = self._save_mobile_report(analysis_result, timestamp)
            logger.info(f"保存移动版报告 (Markdown): {mobile_filepath}")

            return json_filepath
        
        except Exception as e:
            logger.error(f"保存分析报告失败: {e}")
            return ""

    def _save_mobile_report(
        self, analysis_result: Dict[str, Any], timestamp: str
    ) -> str:
        """保存适合窄屏阅读的报告：无参考资料，表格转为逐项列表。"""
        filename = f"mobile_report_{timestamp}.md"
        filepath = os.path.join(self.mobile_reports_dir, filename)
        analysis = analysis_result.get('analysis', '')
        mobile_analysis = self._convert_tables_to_mobile_cards(
            self._remove_references(analysis)
        )

        with open(filepath, 'w', encoding='utf-8') as file:
            file.write("# 全球新闻分析报告（移动版）\n\n")
            file.write(
                f"生成时间: {analysis_result.get('analysis_time', '')}\n\n"
            )
            file.write(mobile_analysis.strip())
            file.write("\n")
        return filepath

    @staticmethod
    def _remove_references(analysis: str) -> str:
        """移除报告末尾完整的新闻参考资料章节。"""
        return re.split(r'\n### 参考资料\s*\n', analysis or '', maxsplit=1)[0]

    @classmethod
    def _convert_tables_to_mobile_cards(cls, markdown: str) -> str:
        """将标准 Markdown 表格改为无横向滚动的逐项阅读格式。"""
        lines = markdown.splitlines()
        output: List[str] = []
        index = 0

        while index < len(lines):
            if (
                index + 1 < len(lines)
                and cls._is_table_row(lines[index])
                and cls._is_table_delimiter(lines[index + 1])
            ):
                headers = cls._split_table_row(lines[index])
                index += 2
                rows: List[List[str]] = []
                while index < len(lines) and cls._is_table_row(lines[index]):
                    rows.append(cls._split_table_row(lines[index]))
                    index += 1

                for row in rows:
                    cells = row[:len(headers)] + [''] * max(0, len(headers) - len(row))
                    if not cells or not cells[0]:
                        continue
                    output.append(f"**{headers[0]}：{cells[0]}**")
                    for header, cell in zip(headers[1:], cells[1:]):
                        if cell:
                            output.append(f"- **{header}：** {cell}")
                    output.append("")
                continue

            output.append(lines[index])
            index += 1

        return "\n".join(output)

    @staticmethod
    def _is_table_row(line: str) -> bool:
        return line.strip().startswith('|') and line.strip().endswith('|')

    @classmethod
    def _is_table_delimiter(cls, line: str) -> bool:
        if not cls._is_table_row(line):
            return False
        return all(
            re.fullmatch(r':?-{3,}:?', cell.strip()) is not None
            for cell in cls._split_table_row(line)
        )

    @staticmethod
    def _split_table_row(line: str) -> List[str]:
        cells = re.split(r'(?<!\\)\|', line.strip().strip('|'))
        return [cell.strip().replace(r'\|', '|') for cell in cells]
    
    def clean_old_cache(self):
        """清理过期的缓存文件"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_cache_days)
            deleted_count = 0

            for directory in [self.news_cache_dir, self.market_cache_dir]:
                if not os.path.exists(directory):
                    continue
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if not os.path.isfile(filepath):
                        continue
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_mtime < cutoff_date:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.debug(f"删除过期缓存: {filename}")
            
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个过期缓存文件")
        
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    def load_historical_news_cache(self, days: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        加载过去 N 天的新闻缓存，按日期分组返回

        每天只取当天最新的一个缓存文件，每天最多返回 max_per_day 篇文章（按
        priority 降序），避免 token 超限。

        Args:
            days: 往前追溯的天数（不含今天）

        Returns:
            {日期字符串(YYYY-MM-DD): [article_dict, ...]} 的有序字典，
            日期从最近到最远排列
        """
        result: Dict[str, List[Dict[str, Any]]] = {}

        try:
            if not os.path.exists(self.news_cache_dir):
                return result

            today = datetime.now().date()

            # 收集候选文件：{date_obj: [filepath, ...]}
            date_files: Dict[Any, List[str]] = {}
            for filename in os.listdir(self.news_cache_dir):
                if not (filename.startswith('news_') and filename.endswith('.json')):
                    continue
                # 文件名格式: news_YYYYMMDD_HHMMSS.json
                parts = filename[5:-5].split('_')  # 去掉 "news_" 和 ".json"
                if len(parts) < 1:
                    continue
                date_str = parts[0]  # YYYYMMDD
                if len(date_str) != 8:
                    continue
                try:
                    file_date = datetime.strptime(date_str, '%Y%m%d').date()
                except ValueError:
                    continue

                # 只保留今天之前的 N 天
                delta = (today - file_date).days
                if delta < 1 or delta > days:
                    continue

                if file_date not in date_files:
                    date_files[file_date] = []
                date_files[file_date].append(
                    os.path.join(self.news_cache_dir, filename)
                )

            # 每天取最新文件，限制每天最多 25 篇
            max_per_day = self.config.get('storage', {}).get(
                'historical_news_max_per_day', 25
            )
            for file_date in sorted(date_files.keys(), reverse=True):
                # 取当天最新文件（按文件名字典序最大即最新）
                latest_file = sorted(date_files[file_date])[-1]
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    articles = data.get('articles', [])

                    # 按 priority 降序截取
                    articles.sort(
                        key=lambda a: a.get('priority', 0), reverse=True
                    )
                    result[file_date.strftime('%Y-%m-%d')] = articles[:max_per_day]
                    logger.debug(
                        f"加载历史新闻 {file_date}: {len(articles[:max_per_day])} 篇 "
                        f"(来自 {os.path.basename(latest_file)})"
                    )
                except Exception as e:
                    logger.warning(f"读取历史缓存失败 {latest_file}: {e}")

            logger.info(f"加载历史新闻完成，共 {len(result)} 天的数据")
        except Exception as e:
            logger.error(f"加载历史新闻缓存失败: {e}")

        return result

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

