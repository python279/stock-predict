"""市场数据、行业标签和舆情摘要的离线单元测试。"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from data_storage import DataStorage
from market_data_fetcher import MarketDataFetcher
from news_fetcher import Article, NewsFetcher
from oge_disclosure_fetcher import OgeDisclosureFetcher
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

    def test_config_tracks_requested_listed_us_technology_companies(self):
        project_root = Path(__file__).resolve().parent.parent
        expected_us = {"INTC", "NVDA", "TSLA", "AMD", "ORCL"}
        expected_kr = {"005930.KS", "000660.KS"}

        for filename in ("config.yaml", "config.example.yaml"):
            config = yaml.safe_load(
                (project_root / filename).read_text(encoding="utf-8")
            )
            assets = config["markets"]["assets"]
            us_symbols = {
                asset["symbol"]
                for asset in assets
                if asset.get("market") == "US"
            }
            kr_symbols = {
                asset["symbol"]
                for asset in assets
                if asset.get("market") == "KR"
            }

            self.assertTrue(expected_us.issubset(us_symbols))
            self.assertTrue(expected_kr.issubset(kr_symbols))
            self.assertNotIn("SPACEX", us_symbols)
            self.assertIn("ZQ=F", {
                item["symbol"] for item in config["macro"]["market_proxies"]
            })
            self.assertIn("TIP", {
                item["symbol"] for item in config["macro"]["market_proxies"]
            })
            self.assertTrue(config["macro"]["treasury_debt_enabled"])
            self.assertTrue(config["macro"]["pboc_m2_enabled"])
            self.assertEqual(
                config["analysis"]["long_term_timeframes"],
                ["未来5年", "未来10年"],
            )
            self.assertTrue(config["oge_disclosure"]["enabled"])
            self.assertEqual(config["oge_disclosure"]["lookback_days"], 180)
            self.assertTrue(
                config["oge_disclosure"]["api_url"].startswith(
                    "https://extapps2.oge.gov/"
                )
            )
            self.assertTrue(
                any(
                    "278-T" in query
                    for query in config["google_news"]["queries"]
                )
            )


class MacroFetcherTest(unittest.TestCase):
    def test_fed_funds_futures_are_converted_to_implied_rate(self):
        from macro_fetcher import MacroFetcher

        fetcher = MacroFetcher(
            {
                "macro": {
                    "enabled": True,
                    "bls_cpi_enabled": False,
                    "treasury_debt_enabled": False,
                    "pboc_m2_enabled": False,
                    "historical_days": 30,
                    "retry_attempts": 1,
                    "market_proxies": [
                        {
                            "name": "联邦基金期货隐含利率",
                            "symbol": "ZQ=F",
                            "category": "policy_expectation",
                            "unit": "%",
                            "transform": "implied_rate_from_100",
                        }
                    ],
                },
                "fetcher": {"retry_attempts": 1},
            }
        )
        fetcher._fetch_yahoo_rows = Mock(
            return_value=[
                {"date": "2026-07-20", "close": 96.0},
                {"date": "2026-07-21", "close": 96.2},
                {"date": "2026-07-22", "close": 96.5},
            ]
        )

        result = fetcher.fetch_all_macro()

        self.assertEqual(len(result["items"]), 1)
        item = result["items"][0]
        self.assertEqual(item["value"], 3.5)
        self.assertEqual(item["raw_price"], 96.5)
        self.assertEqual(item["change_1d_pts"], -0.3)
        self.assertTrue(item["is_proxy"])
        self.assertFalse(item["is_official"])

    def test_bls_failure_is_recorded_without_stopping_proxies(self):
        from macro_fetcher import MacroFetcher

        fetcher = MacroFetcher(
            {
                "macro": {
                    "enabled": True,
                    "bls_cpi_enabled": True,
                    "treasury_debt_enabled": False,
                    "pboc_m2_enabled": False,
                    "retry_attempts": 1,
                    "market_proxies": [
                        {
                            "name": "通胀保值债ETF(TIP)",
                            "symbol": "TIP",
                            "category": "inflation",
                            "unit": "USD",
                        }
                    ],
                },
                "fetcher": {"retry_attempts": 1},
            }
        )
        fetcher._fetch_bls_cpi = Mock(side_effect=RuntimeError("BLS unavailable"))
        fetcher._fetch_yahoo_rows = Mock(
            return_value=[
                {"date": "2026-07-21", "close": 107.0},
                {"date": "2026-07-22", "close": 107.5},
            ]
        )

        result = fetcher.fetch_all_macro()

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["symbol"], "TIP")
        self.assertTrue(any("BLS CPI" in error for error in result["errors"]))

    def test_treasury_debt_scale_and_trends_are_calculated(self):
        from macro_fetcher import MacroFetcher

        fetcher = MacroFetcher(
            {
                "macro": {
                    "enabled": True,
                    "bls_cpi_enabled": False,
                    "treasury_debt_enabled": True,
                    "pboc_m2_enabled": False,
                    "treasury_history_years": 5,
                    "market_proxies": [],
                }
            }
        )
        fetcher._request_json = Mock(
            return_value={
                "data": [
                    {
                        "record_date": "2026-08-12",
                        "tot_pub_debt_out_amt": "40000000000000",
                        "debt_held_public_amt": "32000000000000",
                        "intragov_hold_amt": "8000000000000",
                    },
                    {
                        "record_date": "2025-08-12",
                        "tot_pub_debt_out_amt": "38000000000000",
                        "debt_held_public_amt": "30000000000000",
                        "intragov_hold_amt": "8000000000000",
                    },
                    {
                        "record_date": "2021-08-12",
                        "tot_pub_debt_out_amt": "28000000000000",
                        "debt_held_public_amt": "22000000000000",
                        "intragov_hold_amt": "6000000000000",
                    },
                ]
            }
        )

        result = fetcher.fetch_all_macro()

        self.assertEqual(len(result["items"]), 3)
        total_debt = result["items"][0]
        self.assertEqual(total_debt["value"], 40.0)
        self.assertEqual(total_debt["value_date"], "2026-08-12")
        self.assertEqual(total_debt["change_1y_pct"], 5.26)
        self.assertEqual(total_debt["cagr_5y_pct"], 7.39)
        self.assertEqual(total_debt["source"], "U.S. Treasury Fiscal Data")
        self.assertTrue(total_debt["is_official"])

    def test_pboc_m2_reports_produce_balance_and_yoy_trend(self):
        from macro_fetcher import MacroFetcher

        fetcher = MacroFetcher(
            {
                "macro": {
                    "enabled": True,
                    "bls_cpi_enabled": False,
                    "treasury_debt_enabled": False,
                    "pboc_m2_enabled": True,
                    "pboc_report_limit": 2,
                    "market_proxies": [],
                }
            }
        )
        fetcher._discover_pboc_report_urls = Mock(
            return_value=[
                {"url": "https://www.pbc.gov.cn/current.html", "title": "2026年7月金融统计数据报告"},
                {"url": "https://www.pbc.gov.cn/previous.html", "title": "2026年6月金融统计数据报告"},
            ]
        )
        fetcher._request_text = Mock(
            side_effect=[
                "<p>7月末，广义货币（M2）余额329.94万亿元，同比增长8.1%。</p>",
                "<p>6月末，广义货币(M2)余额328.15万亿元，同比增长8.3%。</p>",
            ]
        )

        result = fetcher.fetch_all_macro()

        self.assertEqual(len(result["items"]), 1)
        m2 = result["items"][0]
        self.assertEqual(m2["value"], 329.94)
        self.assertEqual(m2["value_date"], "2026-07")
        self.assertEqual(m2["yoy_pct"], 8.1)
        self.assertEqual(m2["change_1m_pct_pts"], -0.2)
        self.assertEqual(m2["source"], "中国人民银行")
        self.assertTrue(m2["is_official"])

    def test_official_macro_failures_do_not_stop_other_sources(self):
        from macro_fetcher import MacroFetcher

        fetcher = MacroFetcher(
            {
                "macro": {
                    "enabled": True,
                    "bls_cpi_enabled": False,
                    "treasury_debt_enabled": True,
                    "pboc_m2_enabled": True,
                    "market_proxies": [],
                }
            }
        )
        fetcher._fetch_treasury_debt = Mock(
            side_effect=RuntimeError("Treasury unavailable")
        )
        fetcher._fetch_pboc_m2 = Mock(side_effect=RuntimeError("PBOC unavailable"))

        result = fetcher.fetch_all_macro()

        self.assertEqual(result["items"], [])
        self.assertTrue(any("U.S. Treasury Debt" in error for error in result["errors"]))
        self.assertTrue(any("PBOC M2" in error for error in result["errors"]))


class OgeDisclosureFetcherTest(unittest.TestCase):
    def _config(self, directory=None):
        return {
            "oge_disclosure": {
                "enabled": True,
                "lookback_days": 180,
                "max_filings": 5,
                "max_transactions": 20,
                "max_holdings": 20,
                "pdf_cache_dir": str(
                    Path(directory or tempfile.gettempdir()) / "oge_cache"
                ),
                "retry_attempts": 1,
            },
            "fetcher": {"retry_attempts": 1},
        }

    def test_official_api_rows_are_discovered_and_deduplicated(self):
        fetcher = OgeDisclosureFetcher(self._config())
        official_pdf = (
            "https://extapps2.oge.gov/201/Presiden.nsf/PAS+Index/ABC/"
            "$FILE/Trump-2026-278T.pdf"
        )
        fetcher._request_json = Mock(
            return_value={
                "data": [
                    {
                        "type": f"<a href='{official_pdf}'>278 Transaction</a>",
                        "name": "Trump, Donald J",
                        "agency": "Executive Office of the President",
                        "title": "President",
                        "docDate": datetime.now(timezone.utc).isoformat(),
                        "amended": "",
                    },
                    {
                        "type": f"<a href='{official_pdf}'>278 Transaction</a>",
                        "name": "Trump, Donald J",
                        "agency": "Executive Office of the President",
                        "title": "President",
                        "docDate": datetime.now(timezone.utc).isoformat(),
                        "amended": "",
                    },
                    {
                        "type": "<a href='https://example.com/file.pdf'>278 Transaction</a>",
                        "name": "Another Official",
                        "docDate": datetime.now(timezone.utc).isoformat(),
                    },
                ]
            }
        )

        filings = fetcher._discover_filings()

        self.assertEqual(len(filings), 1)
        self.assertEqual(filings[0]["form_type"], "278-T")
        self.assertEqual(filings[0]["source_url"], official_pdf)
        self.assertTrue(filings[0]["is_official"])

    def test_278t_text_parses_ranges_and_discretionary_flag(self):
        fetcher = OgeDisclosureFetcher(self._config())
        filing = {
            "filing_id": "filing-1",
            "source_url": "https://extapps2.oge.gov/example.pdf",
        }
        pages = [
            {
                "page": 3,
                "text": (
                    "1 APPLE INC COMMON STOCK DISCRETIONARY ORDER "
                    "purchase 6/18/2026 No $1,000,001 - $5,000,000\n"
                    "2 META PLATFORMS INC CLASS A sale 6/18/2026 Yes "
                    "$500,001 - $1,000,000\n"
                    "3 OCR FALSE DATE INC purchase 3/23/2028 No "
                    "$15,001 - $50,000"
                ),
                "tables": [],
            }
        ]

        items = fetcher._parse_transactions(pages, filing)

        self.assertEqual(len(items), 2)
        apple = next(
            item for item in items if item["asset_description"].startswith("APPLE")
        )
        self.assertEqual(apple["transaction_type"], "Purchase")
        self.assertEqual(apple["transaction_date"], "2026-06-18")
        self.assertEqual(apple["amount_min_usd"], 1000001)
        self.assertEqual(apple["amount_max_usd"], 5000000)
        self.assertTrue(apple["discretionary_order"])
        meta = next(
            item for item in items if item["asset_description"].startswith("META")
        )
        self.assertTrue(meta["late_filing"])

    def test_annual_table_only_keeps_verifiable_securities(self):
        fetcher = OgeDisclosureFetcher(self._config())
        pages = [
            {
                "page": 10,
                "text": "Schedule A",
                "tables": [
                    [
                        [
                            "Asset Description",
                            "End of Reporting Period Value",
                            "Income Type",
                            "Income Amount",
                        ],
                        [
                            "ACME CORP COMMON STOCK",
                            "$250,001 - $500,000",
                            "Dividends",
                            "$5,001 - $15,000",
                        ],
                        [
                            "Personal residence",
                            "$1,000,001 - $5,000,000",
                            "",
                            "",
                        ],
                    ]
                ],
            }
        ]
        filing = {
            "filing_id": "annual-1",
            "source_url": "https://extapps2.oge.gov/annual.pdf",
            "report_year": 2026,
        }

        holdings = fetcher._parse_holdings(pages, filing)

        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["asset_description"], "ACME CORP COMMON STOCK")
        self.assertEqual(holdings[0]["value_range"], "$250,001 - $500,000")
        self.assertEqual(holdings[0]["source"], "OGE Form 278e Annual")

    def test_annual_text_rows_are_parsed_without_expensive_table_extraction(self):
        fetcher = OgeDisclosureFetcher(self._config())
        pages = [
            {
                "page": 7,
                "text": (
                    "Part 6: Other Assets and Income\n"
                    "# Description EIF Value Income Type Income Amount\n"
                    "17 SPDR SERIES TRUST S&P 500 ETF Yes "
                    "$1,000,001 - $5,000,000 DIVIDEND $50,001 - $100,000\n"
                    "18 BALL CORP SENIOR UNSECURED NOTES DUE 03/15/2028 "
                    "6.875 N/A $5,000,001 - $25,000,000 INTEREST "
                    "$15,001 - $50,000"
                ),
                "tables": [],
            }
        ]
        filing = {
            "filing_id": "annual-text",
            "source_url": "https://extapps2.oge.gov/annual.pdf",
            "report_year": 2026,
        }

        holdings = fetcher._parse_holdings(pages, filing)

        self.assertEqual(len(holdings), 2)
        etf = next(item for item in holdings if "SPDR" in item["asset_description"])
        bond = next(item for item in holdings if "BALL CORP" in item["asset_description"])
        self.assertEqual(etf["asset_type"], "Fund/ETF")
        self.assertEqual(etf["income_type"], "Dividend")
        self.assertEqual(etf["income_amount"], "$50,001 - $100,000")
        self.assertEqual(etf["value_max_usd"], 5000000)
        self.assertEqual(bond["asset_type"], "Bond/Note")
        self.assertEqual(bond["value_max_usd"], 25000000)
        self.assertNotIn("SPDR", bond["asset_description"])

    def test_scanned_transaction_page_uses_optional_ocr(self):
        fetcher = OgeDisclosureFetcher(self._config())
        fetcher.ocr_enabled = True
        fake_page = Mock()
        fake_page.extract_text.return_value = ""
        fake_pdf = Mock()
        fake_pdf.pages = [fake_page]
        fake_context = Mock()
        fake_context.__enter__ = Mock(return_value=fake_pdf)
        fake_context.__exit__ = Mock(return_value=False)
        fetcher._ocr_pdf_page = Mock(
            return_value=(
                "1 APPLE INC purchase 6/18/2026 No "
                "$1,000,001 - $5,000,000"
            )
        )

        with patch("oge_disclosure_fetcher.pdfplumber.open", return_value=fake_context):
            pages, total = fetcher._extract_pdf_pages(
                b"%PDF mocked", allow_ocr=True
            )

        self.assertEqual(total, 1)
        self.assertTrue(pages[0]["ocr"])
        self.assertIn("APPLE INC", pages[0]["text"])

    def test_failure_uses_last_successful_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            fetcher = OgeDisclosureFetcher(self._config(directory))
            fetcher.cache_dir.mkdir(parents=True, exist_ok=True)
            cached = fetcher._empty_result()
            cached["data_as_of"] = "2026-06-18"
            cached["filings"] = [{"form_type": "278-T"}]
            fetcher._save_snapshot(cached)
            fetcher._discover_filings = Mock(
                side_effect=RuntimeError("OGE unavailable")
            )

            result = fetcher.fetch_all_disclosures([])

            self.assertTrue(result["stale_cache"])
            self.assertEqual(result["data_as_of"], "2026-06-18")
            self.assertTrue(any("使用截至" in error for error in result["errors"]))

    def test_pdf_without_text_layer_is_kept_as_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            fetcher = OgeDisclosureFetcher(self._config(directory))
            filing = {
                "filing_id": "scan-1",
                "form_type": "278-T",
                "report_year": None,
                "received_date": "2026-08-22",
                "name": "Trump, Donald J",
                "title": "President",
                "agency": "Executive Office of the President",
                "source": "U.S. Office of Government Ethics",
                "source_url": "https://extapps2.oge.gov/scanned.pdf",
                "is_official": True,
                "parse_status": "pending",
                "amended": False,
            }
            pdf_path = Path(directory) / "scanned.pdf"
            pdf_path.write_bytes(b"%PDF mocked")
            fetcher._discover_filings = Mock(return_value=[filing])
            fetcher._download_pdf = Mock(return_value=(b"%PDF mocked", pdf_path))
            fetcher._extract_pdf_pages = Mock(return_value=([], 3))

            result = fetcher.fetch_all_disclosures([])

            self.assertEqual(result["filings"][0]["parse_status"], "metadata_only")
            self.assertIn("待人工复核", result["filings"][0]["parse_note"])
            self.assertEqual(result["transactions"], [])

    def test_background_news_requires_trump_disclosure_and_trusted_source(self):
        fetcher = OgeDisclosureFetcher(self._config())
        articles = [
            Article(
                title="Trump files new 278-T stock trade disclosure",
                description="Official filing details",
                content="",
                url="https://www.cnbc.com/example",
                source="CNBC",
                published_at=datetime.now(timezone.utc),
            ),
            Article(
                title="Trump files new stock trade disclosure",
                description="Unverified",
                content="",
                url="https://example.com/untrusted",
                source="Unknown Blog",
                published_at=datetime.now(timezone.utc),
            ),
        ]

        background = fetcher._filter_background_news(articles)

        self.assertEqual(len(background), 1)
        self.assertEqual(background[0]["source"], "CNBC")
        self.assertIn("不得替代", background[0]["note"])


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
    def test_market_cache_serializes_oge_disclosure_data(self):
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
            disclosure = {
                "filings": [{"form_type": "278-T"}],
                "transactions": [{"asset_description": "APPLE INC"}],
                "holdings": [],
            }

            path = storage.save_market_cache({}, {}, disclosure)
            saved = json.loads(Path(path).read_text(encoding="utf-8"))

            self.assertEqual(saved["disclosure_data"], disclosure)

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
                    "articles_count": 50,
                    "commodities_count": 8,
                    "market_items_count": 19,
                    "macro_items_count": 4,
                    "analysis": (
                        "### 1. 今日决策摘要\n\n"
                        "| 行业 | 当前状态 | 风险 |\n"
                        "| :--- | :--- | :--- |\n"
                        "| **科技** | 回调 | 高波动 |\n\n"
                        "### 2. 风险监测\n\n"
                        "*   **已发生事实/信号**: - 冲突升级；<br>- 制裁升级\n"
                        "*   **当前应对**: 保持谨慎。\n\n"
                        "#### 2.1 政策传导\n\n"
                        "| 事件 | 影响 |\n"
                        "| --- | --- |\n"
                        "| 加息预期 | - **科技** (估值承压)；<br>- **消费** (信贷承压) |\n\n"
                        "** 香港市场：**\n\n"
                        "#### 7.3 未来5年、未来10年结构性资产情景\n\n"
                        "| 排名与候选 | 基准情景名义/实际CAGR | 乐观/悲观CAGR | 核心驱动 | 主要风险与失效条件 |\n"
                        "| --- | --- | --- | --- | --- |\n"
                        "| 1. 电力电网 | 8%-12% / 5%-9% | 15%-20% / -5%-0% | AI用电增长 | 资本开支下调 |\n\n"
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
            desktop_report = next((root / "reports").glob("*.md"))
            desktop_content = desktop_report.read_text(encoding="utf-8")

            self.assertIn("# 全球新闻分析报告（移动版）", content)
            self.assertIn("> 数据概览：新闻 50 篇；大宗商品 8 项；市场行情 19 项；宏观指标 4 项", content)
            self.assertIn("#### 行业：科技", content)
            self.assertIn("- **当前状态：** 回调", content)
            self.assertIn(
                "- **已发生事实/信号：**\n  - 冲突升级\n  - 制裁升级",
                content,
            )
            self.assertIn("- **当前应对：** 保持谨慎。", content)
            self.assertIn("##### 事件：加息预期", content)
            self.assertIn(
                "- **影响：**\n  - **科技** (估值承压)\n  - **消费** (信贷承压)",
                content,
            )
            self.assertIn("**香港市场：**", content)
            self.assertIn("#### 7.3 未来5年、未来10年结构性资产情景", content)
            self.assertIn("##### 排名与候选：1. 电力电网", content)
            self.assertIn("- **核心驱动：** AI用电增长", content)
            self.assertNotIn("**行业：**科技**", content)
            self.assertNotIn("*   ", content)
            self.assertNotIn("<br", content)
            self.assertNotIn("** 香港市场", content)
            self.assertNotIn("| :---", content)
            self.assertNotIn("参考资料", content)
            self.assertNotIn("不应出现在移动版的新闻", content)
            self.assertIn(
                "本报告仅为基于公开信息的情景分析，不构成投资建议。",
                content,
            )
            self.assertIn(
                "| 加息预期 | **科技** (估值承压)；**消费** (信贷承压) |",
                desktop_content,
            )
            self.assertIn("**香港市场：**", desktop_content)
            self.assertIn(
                "| 1）电力电网 | 8%-12% / 5%-9% | 15%-20% / -5%-0% |",
                desktop_content,
            )
            self.assertNotIn("<br", desktop_content)
            self.assertNotIn("** 香港市场", desktop_content)
            email_html = EmailSender({})._markdown_to_html(desktop_content)
            self.assertIn("<table>", email_html)
            self.assertIn("未来5年、未来10年结构性资产情景", email_html)


class IndustryReportPromptTest(unittest.TestCase):
    def test_prompt_requires_data_backed_industry_matrix(self):
        analyzer = object.__new__(LLMAnalyzer)
        analyzer.analysis_config = {
            "focus_areas": [],
            "include_predictions": True,
            "include_a_share_analysis": True,
            "short_term_timeframes": ["未来5个交易日", "未来1-4周"],
            "include_long_term_outlook": True,
            "long_term_timeframes": ["未来5年", "未来10年"],
            "us_market_focus": ["科技/AI/半导体与存储"],
        }

        prompt = analyzer._build_system_prompt()

        self.assertIn("固定 7 个", prompt)
        self.assertIn("A股：行业趋势与科技急跌预警", prompt)
        self.assertIn("美股：行业与风险偏好", prompt)
        self.assertIn("科技急跌风险监测", prompt)
        self.assertIn("美国政治热点与全球传导", prompt)
        self.assertIn("证据不足", prompt)
        self.assertIn("已提供的美国科技公司个股行情", prompt)
        self.assertIn("相对 QQQ/SMH 强弱", prompt)
        self.assertIn("SpaceX 等未上市公司的新闻", prompt)
        self.assertIn("全球流动性、财政债务与利率数据", prompt)
        self.assertIn("美国联邦债务", prompt)
        self.assertIn("中国M2", prompt)
        self.assertIn("韩国科技股", prompt)
        self.assertIn("7.3 未来5年、未来10年结构性资产情景", prompt)
        self.assertIn("基准情景名义/实际CAGR", prompt)
        self.assertIn("不得称为“必然疯涨”", prompt)
        self.assertIn("主要风险与失效条件", prompt)
        self.assertIn("OGE 278-T/278e", prompt)
        self.assertIn("唯一一级证据", prompt)
        self.assertIn("DISCRETIONARY ORDER", prompt)
        self.assertIn("不是跟单信号", prompt)
        self.assertIn("最迟45天申报", prompt)
        self.assertIn("2.2 特朗普 OGE 官方交易与持仓披露", prompt)
        self.assertIn("本子节必须始终输出", prompt)
        self.assertIn("OGE文件已发现，但交易明细未可靠解析", prompt)
        self.assertEqual(
            [line for line in prompt.splitlines() if line.startswith("### ")],
            [
                "### 1. 今日决策摘要",
                "### 2. 关键风险与黑天鹅监测",
                "### 3. 宏观与跨资产传导",
                "### 4. A股：行业趋势与科技急跌预警",
                "### 5. 美股：行业与风险偏好",
                "### 6. 港股与日股：联动观察",
                "### 7. 未来路径与执行清单",
            ],
        )

        user_prompt = analyzer._build_user_prompt("测试数据")
        self.assertNotIn("大宗商品章节必须", user_prompt)
        self.assertIn("未来5年/10年的三情景CAGR", user_prompt)
        self.assertIn("第2.2节必须展示特朗普OGE官方交易", user_prompt)
        self.assertIn("第2.1、第2.2和第5章只能以OGE字段", user_prompt)

    def test_disclosure_summary_separates_official_data_and_news(self):
        analyzer = object.__new__(LLMAnalyzer)
        analyzer.config = {
            "oge_disclosure": {
                "max_prompt_transactions": 5,
                "max_prompt_holdings": 5,
            }
        }
        article = Article(
            title="Market update",
            description="details",
            content="details",
            url="https://example.com/news",
            source="Reuters",
            published_at=datetime.now(timezone.utc),
            region="americas",
        )
        disclosure = {
            "data_as_of": "2026-06-18",
            "stale_cache": False,
            "filings": [
                {
                    "form_type": "278-T",
                    "received_date": "2026-08-22",
                    "report_year": None,
                    "parse_status": "parsed",
                    "source_url": "https://extapps2.oge.gov/file.pdf",
                }
            ],
            "transactions": [
                {
                    "transaction_date": "2026-06-18",
                    "asset_description": "BERKSHIRE HATHAWAY INC",
                    "transaction_type": "Purchase",
                    "amount_range": "$1,000,001 - $5,000,000",
                    "amount_max_usd": 5000000,
                    "late_filing": False,
                    "discretionary_order": True,
                    "source": "OGE Form 278-T",
                }
            ],
            "holdings": [
                {
                    "asset_description": "SPDR S&P 500 ETF",
                    "asset_type": "Fund/ETF",
                    "value_range": "$250,001 - $500,000",
                    "as_of_date": "2026",
                    "parse_confidence": "medium",
                    "source": "OGE Form 278e Annual",
                }
            ],
            "background_news": [
                {
                    "published_at": "2026-08-22",
                    "source": "CNBC",
                    "title": "Trump disclosed securities trades",
                    "url": "https://www.cnbc.com/example",
                }
            ],
            "errors": [],
        }

        summary = analyzer._prepare_news_summary(
            [article], {}, {}, {}, {}, {}, disclosure
        )

        self.assertIn("【特朗普 OGE 官方财务披露】", summary)
        self.assertIn("BERKSHIRE HATHAWAY INC", summary)
        self.assertIn("OGE Form 278e Annual", summary)
        self.assertIn("可信媒体背景，仅辅助解释", summary)
        self.assertIn("并非实时或完整持仓", summary)

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
