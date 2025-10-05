"""
新闻抓取模块
支持多种新闻源（News API、RSS Feed）
"""

import requests
import feedparser
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import time
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)


class Article:
    """新闻文章数据类"""
    
    def __init__(
        self,
        title: str,
        description: str,
        content: str,
        url: str,
        source: str,
        published_at: datetime,
        region: str = "unknown",
        priority: int = 3
    ):
        self.title = title
        self.description = description
        self.content = content
        self.url = url
        self.source = source
        self.published_at = published_at
        self.region = region
        self.priority = priority
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """生成文章唯一ID"""
        unique_str = f"{self.url}{self.title}{self.source}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'content': self.content,
            'url': self.url,
            'source': self.source,
            'published_at': self.published_at.isoformat(),
            'region': self.region,
            'priority': self.priority
        }
    
    def __repr__(self) -> str:
        return f"Article(title='{self.title[:50]}...', source='{self.source}')"


class NewsFetcher:
    """新闻抓取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化新闻抓取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.fetcher_config = config.get('fetcher', {})
        self.content_filter = config.get('content_filter', {})
        self.timeout = self.fetcher_config.get('timeout_seconds', 30)
        self.retry_attempts = self.fetcher_config.get('retry_attempts', 3)
        self.retry_delay = self.fetcher_config.get('retry_delay_seconds', 5)
        self.user_agent = self.fetcher_config.get('user_agent', 'NewsAnalyzer/1.0')
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def fetch_all_news(self) -> List[Article]:
        """
        从所有配置的新闻源抓取新闻
        
        Returns:
            文章列表
        """
        all_articles = []
        
        # 从 News API 抓取
        if self.config.get('news_api', {}).get('enabled', False):
            try:
                articles = self.fetch_from_newsapi()
                all_articles.extend(articles)
                logger.info(f"从 News API 抓取到 {len(articles)} 篇文章")
            except Exception as e:
                logger.error(f"从 News API 抓取失败: {e}")
        
        # 从 RSS Feed 抓取
        if self.config.get('rss_feeds', {}).get('enabled', False):
            try:
                articles = self.fetch_from_rss()
                all_articles.extend(articles)
                logger.info(f"从 RSS Feed 抓取到 {len(articles)} 篇文章")
            except Exception as e:
                logger.error(f"从 RSS Feed 抓取失败: {e}")
        
        # 去重
        unique_articles = self._deduplicate_articles(all_articles)
        logger.info(f"去重后共 {len(unique_articles)} 篇文章")
        
        # 过滤和排序
        filtered_articles = self._filter_articles(unique_articles)
        logger.info(f"过滤后共 {len(filtered_articles)} 篇文章")
        
        # 限制数量
        max_articles = self.fetcher_config.get('max_articles', 50)
        final_articles = filtered_articles[:max_articles]
        
        return final_articles
    
    def fetch_from_newsapi(self) -> List[Article]:
        """
        从 News API 抓取新闻
        
        Returns:
            文章列表
        """
        articles = []
        newsapi_config = self.config.get('news_api', {})
        api_key = newsapi_config.get('api_key', '')
        
        if not api_key or api_key.startswith('your-'):
            logger.warning("News API key 未配置，跳过")
            return articles
        
        countries = newsapi_config.get('countries', [])
        categories = newsapi_config.get('categories', [])
        page_size = newsapi_config.get('page_size', 10)
        max_per_country = newsapi_config.get('max_articles_per_country', 5)
        
        base_url = "https://newsapi.org/v2/top-headlines"
        
        for country in countries:
            for category in categories:
                try:
                    params = {
                        'apiKey': api_key,
                        'country': country,
                        'category': category,
                        'pageSize': page_size
                    }
                    
                    response = self._make_request(base_url, params=params)
                    
                    if response and response.get('status') == 'ok':
                        news_articles = response.get('articles', [])
                        
                        for item in news_articles[:max_per_country]:
                            article = self._parse_newsapi_article(item, country)
                            if article:
                                articles.append(article)
                    
                    # 避免请求过快
                    time.sleep(0.5)
                
                except Exception as e:
                    logger.error(f"抓取 News API ({country}/{category}) 失败: {e}")
                    continue
        
        return articles
    
    def fetch_from_rss(self) -> List[Article]:
        """
        从 RSS Feed 抓取新闻
        
        Returns:
            文章列表
        """
        articles = []
        rss_sources = self.config.get('rss_feeds', {}).get('sources', [])
        
        for source in rss_sources:
            try:
                source_name = source.get('name', 'Unknown')
                feed_url = source.get('url', '')
                region = source.get('region', 'unknown')
                priority = source.get('priority', 3)
                
                if not feed_url:
                    continue
                
                logger.info(f"正在抓取 RSS: {source_name}")
                
                # 使用 feedparser 解析 RSS
                feed = feedparser.parse(
                    feed_url,
                    agent=self.user_agent,
                    request_headers={'User-Agent': self.user_agent}
                )
                
                if feed.bozo:
                    logger.warning(f"RSS 解析警告 ({source_name}): {feed.bozo_exception}")
                
                for entry in feed.entries[:10]:  # 限制每个源的数量
                    article = self._parse_rss_entry(entry, source_name, region, priority)
                    if article:
                        articles.append(article)
                
                # 避免请求过快
                time.sleep(0.3)
            
            except Exception as e:
                logger.error(f"抓取 RSS ({source.get('name')}) 失败: {e}")
                continue
        
        return articles
    
    def _parse_newsapi_article(self, item: Dict[str, Any], country: str) -> Optional[Article]:
        """解析 News API 文章"""
        try:
            # 安全获取字段，处理 None 值
            title = (item.get('title') or '').strip()
            description = (item.get('description') or '').strip()
            content = (item.get('content') or '').strip()
            url = (item.get('url') or '').strip()
            source_name = item.get('source', {}).get('name', 'Unknown')
            published_at_str = item.get('publishedAt', '')
            
            if not title or not url:
                return None
            
            # 解析发布时间（确保带时区）
            try:
                published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
            except:
                # 使用 UTC 时区
                published_at = datetime.now(timezone.utc)
            
            # 映射国家代码到区域
            region_map = {
                'us': 'americas',
                'gb': 'europe',
                'de': 'europe',
                'fr': 'europe',
                'ru': 'russia',
                'sg': 'asia',
                'th': 'asia',
                'id': 'asia',
                'jp': 'asia',
                'in': 'asia'
            }
            region = region_map.get(country, 'global')
            
            return Article(
                title=title,
                description=description,
                content=content or description,
                url=url,
                source=source_name,
                published_at=published_at,
                region=region,
                priority=4
            )
        
        except Exception as e:
            logger.error(f"解析 News API 文章失败: {e}")
            return None
    
    def _parse_rss_entry(
        self,
        entry: Any,
        source: str,
        region: str,
        priority: int
    ) -> Optional[Article]:
        """解析 RSS 条目"""
        try:
            # 安全获取字段，处理 None 值
            title = (entry.get('title') or '').strip()
            summary = entry.get('summary') or entry.get('description') or ''
            description = (summary if isinstance(summary, str) else '').strip()
            
            # 处理 content 字段
            content_list = entry.get('content', [])
            if content_list and isinstance(content_list, list) and len(content_list) > 0:
                content = (content_list[0].get('value') or description).strip()
            else:
                content = description
            
            url = (entry.get('link') or '').strip()
            
            if not title or not url:
                return None
            
            # 解析发布时间（确保带时区信息）
            published_at = datetime.now(timezone.utc)
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    # RSS 时间通常是 UTC，添加时区信息
                    published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except:
                    pass
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                try:
                    # RSS 时间通常是 UTC，添加时区信息
                    published_at = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                except:
                    pass
            
            return Article(
                title=title,
                description=description,
                content=content,
                url=url,
                source=source,
                published_at=published_at,
                region=region,
                priority=priority
            )
        
        except Exception as e:
            logger.error(f"解析 RSS 条目失败: {e}")
            return None
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        发起HTTP请求（带重试机制）
        
        Args:
            url: 请求URL
            params: 请求参数
        
        Returns:
            响应JSON或None
        """
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"请求最终失败: {url}")
                    return None
        
        return None
    
    def _deduplicate_articles(self, articles: List[Article]) -> List[Article]:
        """去重文章（基于ID）"""
        seen_ids = set()
        unique_articles = []
        
        for article in articles:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique_articles.append(article)
        
        return unique_articles
    
    def _filter_articles(self, articles: List[Article]) -> List[Article]:
        """
        过滤文章
        
        Args:
            articles: 文章列表
        
        Returns:
            过滤后的文章列表
        """
        filtered = []
        min_length = self.content_filter.get('min_content_length', 100)
        exclude_keywords = self.content_filter.get('exclude_keywords', [])
        include_keywords = self.content_filter.get('include_keywords', [])
        
        for article in articles:
            # 检查最小长度
            if len(article.content) < min_length:
                continue
            
            # 检查排除关键词
            text_lower = f"{article.title} {article.description} {article.content}".lower()
            if any(keyword.lower() in text_lower for keyword in exclude_keywords):
                continue
            
            # 优先级提升（包含关键词）
            if any(keyword.lower() in text_lower for keyword in include_keywords):
                article.priority += 1
            
            filtered.append(article)
        
        # 按优先级和时间排序
        filtered.sort(key=lambda x: (x.priority, x.published_at), reverse=True)
        
        return filtered

