"""轻量行业舆情摘要。

默认仅对已抓取的权威新闻做透明的标题情绪统计。可选的东方财富股吧采集
必须显式配置代码并启用，失败时只在结果中记录，不影响日报。
"""

import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class SentimentFetcher:
    """以新闻为主、可选股吧标题为补充的行业舆情聚合器。"""

    POSITIVE_WORDS = {
        "surge", "gain", "growth", "beat", "upgrade", "recovery", "stimulus",
        "上涨", "增长", "利好", "回暖", "突破", "扩产", "增持",
    }
    NEGATIVE_WORDS = {
        "fall", "drop", "plunge", "cut", "downgrade", "risk", "default",
        "sanction", "crisis", "decline", "下跌", "暴跌", "利空", "风险",
        "减持", "制裁", "危机",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        sentiment_config = config.get("sentiment", {})
        fetcher_config = config.get("fetcher", {})
        self.enabled = sentiment_config.get("enabled", True)
        self.guba_enabled = sentiment_config.get("guba_enabled", False)
        self.guba_codes = sentiment_config.get("guba_codes", {})
        self.timeout = sentiment_config.get(
            "timeout_seconds", fetcher_config.get("timeout_seconds", 30)
        )
        self.retry_attempts = sentiment_config.get(
            "retry_attempts", fetcher_config.get("retry_attempts", 3)
        )
        self.retry_delay = sentiment_config.get(
            "retry_delay_seconds", fetcher_config.get("retry_delay_seconds", 5)
        )
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": fetcher_config.get("user_agent", "NewsAnalyzer/1.0")}
        )

    def fetch(self, articles: List[Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """汇总新闻、行情与可选股吧标题，返回行业级而非伪精确情绪分数。"""
        result: Dict[str, Any] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
            "errors": [],
        }
        if not self.enabled:
            return result

        grouped: Dict[str, List[str]] = defaultdict(list)
        for article in articles:
            for tag in getattr(article, "tags", []):
                if tag in {"tech", "finance", "consumer"}:
                    grouped[tag].append(article.title)

        market_by_sector: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in market_data.get("items", []):
            sector = item.get("sector")
            if sector in {"tech", "finance", "consumer"}:
                market_by_sector[sector].append(item)

        for sector in ("tech", "finance", "consumer"):
            titles = grouped[sector]
            guba_titles = self._fetch_guba_titles(sector, result["errors"])
            all_titles = titles + guba_titles
            positive, negative = self._count_sentiment(all_titles)
            result["items"].append(
                {
                    "sector": sector,
                    "sample_size": len(all_titles),
                    "news_sample_size": len(titles),
                    "guba_sample_size": len(guba_titles),
                    "positive_titles": positive,
                    "negative_titles": negative,
                    "sentiment": self._sentiment_label(positive, negative),
                    "topics": self._top_topics(all_titles),
                    "sample_titles": all_titles[:5],
                    "market_context": [
                        {
                            "name": item.get("name"),
                            "trend_signal": item.get("trend_signal"),
                            "change_5d_pct": item.get("change_5d_pct"),
                        }
                        for item in market_by_sector[sector]
                    ],
                }
            )
        return result

    def _fetch_guba_titles(self, sector: str, errors: List[str]) -> List[str]:
        if not self.guba_enabled:
            return []
        titles: List[str] = []
        for code in self.guba_codes.get(sector, []):
            try:
                response = self._request(f"https://guba.eastmoney.com/list,{code}.html")
                soup = BeautifulSoup(response.text, "html.parser")
                for node in soup.select("a[href*='news,'], a[href*='read,']"):
                    title = node.get_text(" ", strip=True)
                    if title:
                        titles.append(title)
                    if len(titles) >= 10:
                        break
            except requests.RequestException as exc:
                errors.append(f"股吧 {code}: {exc}")
            time.sleep(0.2)
        return titles[:10]

    def _request(self, url: str) -> requests.Response:
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == self.retry_attempts - 1:
                    raise
                time.sleep(self.retry_delay)
        raise requests.RequestException("无法获取舆情数据")

    @classmethod
    def _count_sentiment(cls, titles: List[str]) -> tuple[int, int]:
        positive = negative = 0
        for title in titles:
            lower = title.lower()
            positive += sum(word.lower() in lower for word in cls.POSITIVE_WORDS)
            negative += sum(word.lower() in lower for word in cls.NEGATIVE_WORDS)
        return positive, negative

    @staticmethod
    def _sentiment_label(positive: int, negative: int) -> str:
        if positive + negative == 0:
            return "中性/样本不足"
        if positive >= negative * 1.5:
            return "偏正面"
        if negative >= positive * 1.5:
            return "偏负面"
        return "分歧"

    @staticmethod
    def _top_topics(titles: List[str]) -> List[str]:
        tokens = Counter()
        for title in titles:
            tokens.update(
                token.lower()
                for token in re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}", title)
            )
        return [word for word, _ in tokens.most_common(5)]
