"""市场数据、行业标签和舆情摘要的离线单元测试。"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from data_storage import DataStorage
from market_data_fetcher import MarketDataFetcher
from news_fetcher import Article, NewsFetcher
from sentiment_fetcher import SentimentFetcher
from llm_analyzer import LLMAnalyzer
from email_sender import EmailSender


class MarketDataFetcherTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "markets": {
                "enabled": True,
                "historical_days": 60,
                "assets": [
                    {
                        "name": "科技ETF",
                        "symbol": "XLK",
                        "provider": "yahoo",
                        "market": "US",
                        "sector": "tech",
                    }
                ],
            },
            "fetcher": {"retry_attempts": 1},
        }

    def test_yahoo_data_is_normalized_to_trend_snapshot(self):
        fetcher = MarketDataFetcher(self.config)
        rows = [
            {"date": f"2026-01-{index + 1:02d}", "close": 100 + index, "volume": 1000}
            for index in range(60)
        ]
        fetcher._fetch_yahoo_rows = Mock(return_value=rows)

        result = fetcher.fetch_all_markets()

        self.assertEqual(len(result["items"]), 1)
        item = result["items"][0]
        self.assertEqual(item["name"], "科技ETF")
        self.assertEqual(item["price"], 159)
        self.assertEqual(item["trend_signal"], "上升趋势")
        self.assertEqual(item["price_date"], "2026-01-60")
        self.assertIsNotNone(item["drawdown_20d_pct"])
        self.assertIsNotNone(item["volatility_20d_pct"])

    def test_tencent_quote_is_parsed_before_eastmoney_fallback(self):
        fetcher = MarketDataFetcher(self.config)
        values = [""] * 31
        values[3] = "4529.10"
        values[6] = "302629886"
        values[30] = "20260717161408"
        response = Mock(content=f'v_sh000300="{"~".join(values)}";'.encode("gbk"))
        response.raise_for_status = Mock()
        fetcher.session.get = Mock(return_value=response)

        quote = fetcher._fetch_tencent_quote("sh000300")

        self.assertEqual(quote["price"], 4529.10)
        self.assertEqual(quote["volume"], 302629886)
        self.assertEqual(quote["date"], "2026-07-17")

    def test_tech_risk_requires_multiple_observable_signals(self):
        items = [
            {"market": "CN", "sector": "broad", "change_5d_pct": -1},
            {
                "market": "CN",
                "sector": "tech",
                "change_1d_pct": -2,
                "change_5d_pct": -5,
                "change_20d_pct": -10,
                "drawdown_20d_pct": -11,
                "price": 80,
                "ma20": 90,
                "ma60": 100,
                "volume_ratio_20d": 1.5,
            },
        ]

        MarketDataFetcher._add_sector_risk_assessments(items)

        self.assertEqual(items[1]["risk_level"], "高")
        self.assertGreaterEqual(len(items[1]["risk_signals"]), 3)


class IndustryTagAndSentimentTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "content_filter": {
                "min_content_length": 1,
                "exclude_keywords": [],
                "include_keywords": [],
                "sector_keywords": {
                    "tech": ["AI", "chip"],
                    "finance": ["bank"],
                    "consumer": ["retail"],
                    "policy": ["policy"],
                    "us_politics": ["White House", "election"],
                },
            },
            "fetcher": {"max_articles": 10, "retry_attempts": 1},
            "sentiment": {"enabled": True, "guba_enabled": False},
        }

    def test_article_can_have_multiple_industry_tags(self):
        article = Article(
            title="White House AI chip policy supports retail banks",
            description="detail",
            content="detail",
            url="https://example.com/article",
            source="test",
            published_at=datetime.now(timezone.utc),
        )
        filtered = NewsFetcher(self.config)._filter_articles([article])

        self.assertEqual(len(filtered), 1)
        self.assertEqual(
            set(filtered[0].tags),
            {"tech", "finance", "consumer", "policy", "us_politics"},
        )

    def test_sentiment_keeps_sample_size_and_market_context(self):
        article = Article(
            title="AI chip growth accelerates",
            description="detail",
            content="detail",
            url="https://example.com/sentiment",
            source="test",
            published_at=datetime.now(timezone.utc),
            tags=["tech"],
        )
        result = SentimentFetcher(self.config).fetch(
            [article],
            {
                "items": [
                    {
                        "name": "科技ETF",
                        "sector": "tech",
                        "trend_signal": "上升趋势",
                        "change_5d_pct": 3.2,
                    }
                ]
            },
        )
        tech = next(item for item in result["items"] if item["sector"] == "tech")
        self.assertEqual(tech["sample_size"], 1)
        self.assertEqual(tech["sentiment"], "偏正面")
        self.assertEqual(tech["market_context"][0]["name"], "科技ETF")

    def test_newsapi_rate_limit_stops_remaining_requests(self):
        config = {
            "news_api": {
                "api_key": "test-key",
                "countries": ["us", "gb"],
                "categories": ["business", "technology"],
                "page_size": 1,
            },
            "fetcher": {"retry_attempts": 3},
        }
        fetcher = NewsFetcher(config)
        response = Mock(status_code=429, headers={"Retry-After": "60"})
        fetcher.session.get = Mock(return_value=response)

        self.assertEqual(fetcher.fetch_from_newsapi(), [])
        self.assertEqual(fetcher.session.get.call_count, 1)

    def test_gdelt_headline_respects_domain_whitelist(self):
        fetcher = NewsFetcher(self.config)
        article = fetcher._parse_gdelt_article(
            {
                "title": "Chip market gains",
                "url": "https://www.reuters.com/example",
                "domain": "reuters.com",
                "seendate": "20260718T010203Z",
                "sourcecountry": "US",
            }
        )

        self.assertTrue(article.headline_only)
        self.assertTrue(
            fetcher._is_trusted_domain("www.reuters.com", {"reuters.com"})
        )
        self.assertFalse(fetcher._is_trusted_domain("example.com", {"reuters.com"}))

    def test_imf_archive_reads_official_dynamic_api(self):
        config = {
            "imf_archive": {
                "enabled": True,
                "max_records": 2,
                "sectors": ["finance", "policy"],
            },
            "fetcher": {"retry_attempts": 1},
        }
        fetcher = NewsFetcher(config)
        page_response = Mock(status_code=200)
        page_response.text = (
            '<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(
                {
                    "props": {
                        "pageProps": {
                            "page": {
                                "layout": {
                                    "sitecore": {
                                        "route": {"itemId": "archive-item-id"}
                                    }
                                }
                            }
                        }
                    }
                }
            )
            + "</script>"
        )
        page_response.raise_for_status = Mock()
        api_response = Mock(status_code=200)
        api_response.json.return_value = {
            "search": {
                "results": [
                    {
                        "languages": [
                            {
                                "language": {"name": "en"},
                                "mainTitle": {"jsonValue": {"value": "IMF policy update"}},
                                "mainTitleLink": {
                                    "url": "/en/news/articles/2026/07/19/pr-imf-example"
                                },
                            }
                        ],
                        "description": {
                            "jsonValue": {"value": "<p>Official IMF announcement.</p>"}
                        },
                        "fromDateTime": {"jsonValue": {"value": "2026-07-19T09:00:00Z"}},
                    }
                ]
            }
        }
        api_response.raise_for_status = Mock()
        fetcher.session.get = Mock(side_effect=[page_response, api_response])

        articles = fetcher.fetch_from_imf_archive()

        self.assertEqual(len(articles), 1)
        self.assertEqual(fetcher.session.get.call_args_list[1].args[0], "https://www.imf.org/api/oap/news-archive")
        self.assertEqual(articles[0].source, "IMF What's New Archive")
        self.assertEqual(
            articles[0].url,
            "https://www.imf.org/en/news/articles/2026/07/19/pr-imf-example",
        )
        self.assertTrue(articles[0].headline_only)
        self.assertEqual(articles[0].tags, ["finance", "policy"])


class MobileReportStorageTest(unittest.TestCase):
    def test_mobile_report_removes_references_and_converts_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = DataStorage(
                {
                    "storage": {
                        "news_cache_dir": str(root / "news_cache"),
                        "reports_dir": str(root / "reports"),
                        "mobile_reports_dir": str(root / "mobile_reports"),
                        "market_cache_dir": str(root / "market_cache"),
                    }
                }
            )
            storage.save_analysis_report(
                {
                    "analysis_time": "2026-07-19T19:00:00",
                    "analysis": (
                        "### 1. 今日决策摘要\n\n"
                        "| 行业 | 当前状态 | 风险 |\n"
                        "| :--- | :--- | :--- |\n"
                        "| 科技 | 回调 | 高波动 |\n\n"
                        "本报告仅为基于公开信息的情景分析，不构成投资建议。\n\n"
                        "### 参考资料\n\n"
                        "1. **不应出现在移动版的新闻**"
                    ),
                    "regions_covered": [],
                    "all_articles": [],
                }
            )

            mobile_report = next((root / "mobile_reports").glob("*.md"))
            content = mobile_report.read_text(encoding="utf-8")

            self.assertIn("# 全球新闻分析报告（移动版）", content)
            self.assertIn("**行业：科技**", content)
            self.assertIn("- **当前状态：** 回调", content)
            self.assertNotIn("| :---", content)
            self.assertNotIn("参考资料", content)
            self.assertNotIn("不应出现在移动版的新闻", content)


class IndustryReportPromptTest(unittest.TestCase):
    def test_prompt_requires_data_backed_industry_matrix(self):
        analyzer = object.__new__(LLMAnalyzer)
        analyzer.analysis_config = {
            "focus_areas": [],
            "include_predictions": True,
            "include_a_share_analysis": True,
            "short_term_timeframes": ["未来5个交易日", "未来1-4周"],
            "us_market_focus": ["科技/AI/半导体与存储"],
        }

        prompt = analyzer._build_system_prompt()

        self.assertIn("固定 7 个", prompt)
        self.assertIn("A股：行业趋势与科技急跌预警", prompt)
        self.assertIn("美股：行业与风险偏好", prompt)
        self.assertIn("科技急跌风险监测", prompt)
        self.assertIn("美国政治热点与全球传导", prompt)
        self.assertIn("证据不足", prompt)

    def test_report_icons_are_removed_without_affecting_markdown(self):
        cleaned = LLMAnalyzer._strip_report_icons(
            "### 摘要 📈\n\n- 风险 ⚠️：价格上行 → 保持谨慎"
        )

        self.assertEqual(cleaned, "### 摘要 \n\n- 风险 ：价格上行 → 保持谨慎")


class MarkdownFormattingTest(unittest.TestCase):
    def test_latex_arrow_mapping_is_rendered_as_list_text(self):
        html = EmailSender({})._markdown_to_html(
            "    - 今日事实 $军事冲突 \\rightarrow 能源风险$ → 历史模式 → 策略"
        )

        self.assertIn("<li>", html)
        self.assertIn("军事冲突 → 能源风险", html)
        self.assertNotIn(r"\rightarrow", html)
        self.assertNotIn("<code>", html)

    def test_inline_lists_inside_table_cells_are_normalized(self):
        html = EmailSender({})._markdown_to_html(
            "| 风险 | 信号 |\n"
            "| --- | --- |\n"
            "| 中东 | - 冲突升级 - 航道受阻 1. 能源上涨 |"
        )

        self.assertIn("<table>", html)
        self.assertIn("冲突升级；航道受阻；能源上涨", html)
        self.assertNotIn(" - 冲突升级", html)


if __name__ == "__main__":
    unittest.main()
