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
from urllib.parse import urlencode
import hashlib
import json
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """外部新闻源返回 429 时中止该源的后续请求。"""


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
        priority: int = 3,
        tags: Optional[List[str]] = None,
        headline_only: bool = False,
    ):
        self.title = title
        self.description = description
        self.content = content
        self.url = url
        self.source = source
        self.published_at = published_at
        self.region = region
        self.priority = priority
        self.tags = tags or []
        self.headline_only = headline_only
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
            'priority': self.priority,
            'tags': self.tags,
            'headline_only': self.headline_only,
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

        # IMF 不稳定的 RSS 页面改用官网“What's New”归档页尽力抓取。
        if self.config.get('imf_archive', {}).get('enabled', False):
            try:
                articles = self.fetch_from_imf_archive()
                all_articles.extend(articles)
                logger.info(f"从 IMF What's New 归档抓取到 {len(articles)} 篇文章")
            except Exception as e:
                logger.error(f"从 IMF What's New 归档抓取失败: {e}")

        # GDELT 是无需密钥的全球新闻索引，作为 NewsAPI 的免费补充。
        if self.config.get('gdelt', {}).get('enabled', False):
            try:
                articles = self.fetch_from_gdelt()
                all_articles.extend(articles)
                logger.info(f"从 GDELT 抓取到 {len(articles)} 篇文章")
            except Exception as e:
                logger.error(f"从 GDELT 抓取失败: {e}")

        # Google News RSS 无需密钥；只保留已配置的权威媒体名称。
        if self.config.get('google_news', {}).get('enabled', False):
            try:
                articles = self.fetch_from_google_news()
                all_articles.extend(articles)
                logger.info(f"从 Google News RSS 抓取到 {len(articles)} 篇文章")
            except Exception as e:
                logger.error(f"从 Google News RSS 抓取失败: {e}")
        
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
        rate_limited = False
        
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
                except RateLimitError:
                    rate_limited = True
                    logger.warning(
                        "NewsAPI 免费额度或速率已耗尽，停止本轮后续 NewsAPI 请求；"
                        "将继续使用 RSS/GDELT。"
                    )
                    break
                except Exception as e:
                    logger.error(f"抓取 News API ({country}/{category}) 失败: {e}")
                    continue
            if rate_limited:
                break
        
        return articles

    def fetch_from_google_news(self) -> List[Article]:
        """抓取 Google News 搜索 RSS，仅作为权威源 RSS 的补充。"""
        google_config = self.config.get('google_news', {})
        trusted_sources = {
            source.lower() for source in google_config.get('trusted_sources', [])
        }
        articles: List[Article] = []
        for query in google_config.get('queries', []):
            url = "https://news.google.com/rss/search?" + urlencode(
                {
                    'q': query,
                    'hl': google_config.get('language', 'en-US'),
                    'gl': google_config.get('country', 'US'),
                    'ceid': google_config.get('ceid', 'US:en'),
                }
            )
            feed = feedparser.parse(
                url,
                agent=self.user_agent,
                request_headers={'User-Agent': self.user_agent},
            )
            if feed.bozo:
                logger.warning("Google News RSS 解析警告: %s", feed.bozo_exception)
            for entry in feed.entries[:google_config.get('max_records_per_query', 20)]:
                source_data = entry.get('source', {}) or {}
                source_name = (source_data.get('title') or '').strip()
                if trusted_sources and source_name.lower() not in trusted_sources:
                    continue
                article = self._parse_google_news_entry(entry, source_name)
                if article:
                    articles.append(article)
            time.sleep(0.3)
        return articles

    def fetch_from_imf_archive(self) -> List[Article]:
        """通过 IMF 官网归档的动态 JSON 接口抓取官方更新。"""
        archive_config = self.config.get('imf_archive', {})
        url = archive_config.get('url', 'https://www.imf.org/en/whats-new-archive')
        page_html = self._request_html(url)
        if not page_html:
            return []

        item_id = archive_config.get('item_id') or self._imf_archive_item_id(page_html)
        if not item_id:
            logger.warning("未能从 IMF 归档页面识别动态接口 ID，跳过本轮归档抓取")
            return []

        now = datetime.now(timezone.utc)
        lookback_days = archive_config.get('lookback_days', 30)
        data = self._make_request(
            "https://www.imf.org/api/oap/news-archive",
            params={
                'itemLimit': archive_config.get('max_records', 10),
                'language': archive_config.get('language', 'en'),
                'itemId': item_id,
                'startdate': (now - timedelta(days=lookback_days)).date().isoformat(),
                'enddate': now.date().isoformat(),
                'after': '',
            },
        )
        results = (data or {}).get('search', {}).get('results', [])
        max_records = archive_config.get('max_records', 10)
        articles: List[Article] = []
        seen_urls = set()

        for result in results:
            language = next(
                (
                    item for item in result.get('languages', [])
                    if item.get('language', {}).get('name') == archive_config.get('language', 'en')
                ),
                {},
            )
            title = self._nested_value(language, 'mainTitle', 'jsonValue', 'value')
            article_url = self._nested_value(language, 'mainTitleLink', 'url')
            if article_url.startswith('/'):
                article_url = f"https://www.imf.org{article_url}"
            if not title or not article_url or article_url in seen_urls:
                continue
            if not article_url.startswith('https://www.imf.org/'):
                continue

            description_html = self._nested_value(result, 'description', 'jsonValue', 'value')
            description = BeautifulSoup(description_html, 'html.parser').get_text(' ', strip=True)
            published_at = self._parse_imf_datetime(
                self._nested_value(result, 'fromDateTime', 'jsonValue', 'value'),
                now,
            )
            seen_urls.add(article_url)
            articles.append(
                Article(
                    title=title,
                    description=description or "IMF What's New 官网归档标题索引",
                    content=description or title,
                    url=article_url,
                    source="IMF What's New Archive",
                    published_at=published_at,
                    region=archive_config.get('region', 'international'),
                    priority=archive_config.get('priority', 5),
                    tags=archive_config.get('sectors', ['finance', 'policy']),
                    headline_only=True,
                )
            )
            if len(articles) >= max_records:
                break
        return articles

    @staticmethod
    def _imf_archive_item_id(page_html: str) -> str:
        """从 IMF Next.js 页面数据中读取 What's New 归档的组件 ID。"""
        try:
            soup = BeautifulSoup(page_html, 'html.parser')
            data_tag = soup.find('script', id='__NEXT_DATA__')
            data = json.loads(data_tag.string) if data_tag and data_tag.string else {}
            return (
                data.get('props', {})
                .get('pageProps', {})
                .get('page', {})
                .get('layout', {})
                .get('sitecore', {})
                .get('route', {})
                .get('itemId', '')
            )
        except (json.JSONDecodeError, AttributeError, TypeError):
            return ''

    @staticmethod
    def _nested_value(data: Dict[str, Any], *keys: str) -> str:
        """安全读取 IMF 接口的嵌套文本字段。"""
        value: Any = data
        for key in keys:
            if not isinstance(value, dict):
                return ''
            value = value.get(key)
        return value.strip() if isinstance(value, str) else ''

    @staticmethod
    def _parse_imf_datetime(value: str, fallback: datetime) -> datetime:
        """解析 IMF 接口 ISO 时间，缺失时保留本轮抓取时间。"""
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (TypeError, ValueError):
            return fallback

    def fetch_from_gdelt(self) -> List[Article]:
        """从免费 GDELT DOC API 抓取配置化的全球新闻标题。"""
        gdelt_config = self.config.get('gdelt', {})
        queries = gdelt_config.get('queries', [])
        max_records = gdelt_config.get('max_records_per_query', 20)
        timespan = gdelt_config.get('timespan', '1d')
        domains = {
            domain.lower().lstrip('.')
            for domain in gdelt_config.get('trusted_domains', [])
        }
        articles: List[Article] = []

        for query in queries:
            try:
                data = self._make_request(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params={
                        'query': query,
                        'mode': 'ArtList',
                        'format': 'json',
                        'maxrecords': max_records,
                        'timespan': timespan,
                    },
                )
            except RateLimitError as exc:
                logger.warning("GDELT 触发限流，停止本轮后续 GDELT 查询: %s", exc)
                break
            if not data:
                continue

            for item in data.get('articles', []):
                domain = (item.get('domain') or '').lower().lstrip('.')
                if domains and not self._is_trusted_domain(domain, domains):
                    continue
                article = self._parse_gdelt_article(item)
                if article:
                    articles.append(article)
            time.sleep(0.3)

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
                source_tags = source.get('sectors', [])
                
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
                    article = self._parse_rss_entry(
                        entry, source_name, region, priority, source_tags
                    )
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
        priority: int,
        tags: Optional[List[str]] = None,
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
                priority=priority,
                tags=tags,
            )
        
        except Exception as e:
            logger.error(f"解析 RSS 条目失败: {e}")
            return None

    def _parse_gdelt_article(self, item: Dict[str, Any]) -> Optional[Article]:
        """解析 GDELT 标题索引；GDELT 不提供正文，明确标记为标题级数据。"""
        try:
            title = (item.get('title') or '').strip()
            url = (item.get('url') or '').strip()
            if not title or not url:
                return None

            seen_date = (item.get('seendate') or '').strip()
            try:
                published_at = datetime.strptime(
                    seen_date[:14], '%Y%m%dT%H%M%S'
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                published_at = datetime.now(timezone.utc)

            domain = (item.get('domain') or 'GDELT').strip()
            language = (item.get('language') or '').strip()
            description = f"GDELT 标题索引；来源域名：{domain}；语言：{language}"
            return Article(
                title=title,
                description=description,
                content=title,
                url=url,
                source=f"GDELT / {domain}",
                published_at=published_at,
                region=self._gdelt_region(item.get('sourcecountry')),
                priority=4,
                headline_only=True,
            )
        except Exception as e:
            logger.warning(f"解析 GDELT 文章失败: {e}")
            return None

    def _parse_google_news_entry(
        self, entry: Any, source_name: str
    ) -> Optional[Article]:
        """解析 Google News RSS 聚合标题，不将其摘要伪装成原文正文。"""
        try:
            title = (entry.get('title') or '').strip()
            url = (entry.get('link') or '').strip()
            if not title or not url:
                return None
            published_at = datetime.now(timezone.utc)
            if entry.get('published_parsed'):
                published_at = datetime(
                    *entry.published_parsed[:6], tzinfo=timezone.utc
                )
            return Article(
                title=title,
                description=f"Google News RSS 标题索引；来源：{source_name or '未知'}",
                content=title,
                url=url,
                source=f"Google News / {source_name or '未知'}",
                published_at=published_at,
                region='global',
                priority=4,
                headline_only=True,
            )
        except Exception as e:
            logger.warning("解析 Google News RSS 文章失败: %s", e)
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
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    wait_hint = f"，Retry-After={retry_after}" if retry_after else ""
                    raise RateLimitError(f"HTTP 429{wait_hint}")
                response.raise_for_status()
                return response.json()

            except RateLimitError:
                raise
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"请求最终失败: {url}")
                    return None
        
        return None

    def _request_html(self, url: str) -> Optional[str]:
        """请求 HTML 页面；访问限制或单源故障不影响其他新闻源。"""
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 403:
                    logger.warning("IMF 官网拒绝本轮归档请求（HTTP 403），将继续其他新闻源")
                    return None
                if response.status_code == 429:
                    logger.warning("IMF 官网触发限流（HTTP 429），停止本轮归档请求")
                    return None
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "HTML 请求失败 (尝试 %d/%d): %s",
                    attempt + 1,
                    self.retry_attempts,
                    exc,
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
        logger.error("HTML 请求最终失败: %s", url)
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
        sector_keywords = self.content_filter.get('sector_keywords', {})
        
        for article in articles:
            # 检查最小长度
            if len(article.content) < min_length and not article.headline_only:
                continue
            
            # 检查排除关键词
            text_lower = f"{article.title} {article.description} {article.content}".lower()
            if any(keyword.lower() in text_lower for keyword in exclude_keywords):
                continue
            
            # 优先级提升（包含关键词）
            if any(keyword.lower() in text_lower for keyword in include_keywords):
                article.priority += 1
            
            sector_tags = self._tag_article(text_lower, sector_keywords)
            if sector_tags:
                article.tags = sorted(set(article.tags).union(sector_tags))
                # 行业命中比泛关键词更重要，但避免大量标签造成不成比例加权。
                article.priority += min(len(sector_tags), 2)
            
            filtered.append(article)
        
        # 按优先级和时间排序
        filtered.sort(key=lambda x: (x.priority, x.published_at), reverse=True)
        
        return filtered

    @staticmethod
    def _is_trusted_domain(domain: str, trusted_domains: set) -> bool:
        """允许白名单域名及其子域名，避免 GDELT 的聚合噪声进入报告。"""
        return any(domain == item or domain.endswith(f".{item}") for item in trusted_domains)

    @staticmethod
    def _gdelt_region(source_country: Any) -> str:
        """将 GDELT 来源国家粗略映射到现有报告区域。"""
        country = str(source_country or '').upper()
        if country in {'US', 'CA', 'MX', 'BR'}:
            return 'americas'
        if country in {'GB', 'DE', 'FR', 'IT', 'ES', 'EU'}:
            return 'europe'
        if country in {'RU', 'UA'}:
            return 'russia'
        if country in {'CN', 'JP', 'KR', 'SG', 'IN', 'TH', 'ID'}:
            return 'asia'
        if country in {'IR', 'IL', 'SA', 'AE', 'TR'}:
            return 'middle_east'
        return 'global'

    @staticmethod
    def _tag_article(
        text_lower: str, sector_keywords: Dict[str, List[str]]
    ) -> List[str]:
        """按配置关键词为文章打可叠加的行业/政策标签。"""
        tags = []
        for tag, keywords in sector_keywords.items():
            if any(str(keyword).lower() in text_lower for keyword in keywords):
                tags.append(tag)
        return tags

