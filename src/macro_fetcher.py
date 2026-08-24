"""宏观指标抓取：通胀、利率、美国联邦债务与中国广义货币。

官方 CPI 优先尝试 BLS 公开接口；失败时降级为可验证的市场代理
（TIPS ETF、联邦基金期货隐含利率、短端/长端国债收益率），并在输出中
明确标注“数据缺失/市场代理”。财政与货币规模仅使用美国财政部和
中国人民银行官方数据，不得用商业数据冒充官方统计。
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class MacroFetcher:
    """抓取通胀、利率、财政债务与货币供应相关宏观输入。"""

    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    TREASURY_DEBT_URL = (
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
        "v2/accounting/od/debt_to_penny"
    )
    PBOC_REPORT_INDEX_URL = (
        "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html"
    )

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.macro_config = config.get("macro", {})
        fetcher_config = config.get("fetcher", {})
        self.enabled = self.macro_config.get("enabled", False)
        self.timeout = self.macro_config.get(
            "timeout_seconds", fetcher_config.get("timeout_seconds", 20)
        )
        self.retry_attempts = self.macro_config.get(
            "retry_attempts", fetcher_config.get("retry_attempts", 2)
        )
        self.retry_delay = self.macro_config.get(
            "retry_delay_seconds", fetcher_config.get("retry_delay_seconds", 3)
        )
        self.historical_days = self.macro_config.get("historical_days", 90)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": fetcher_config.get("user_agent", "NewsAnalyzer/1.0")}
        )

    def fetch_all_macro(self) -> Dict[str, Any]:
        """抓取宏观指标；单个系列失败不影响其他系列。"""
        result: Dict[str, Any] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
            "errors": [],
        }
        if not self.enabled:
            logger.info("宏观指标抓取未启用")
            return result

        if self.macro_config.get("bls_cpi_enabled", True):
            try:
                items = self._fetch_bls_cpi()
                result["items"].extend(items)
                if not items:
                    result["errors"].append("BLS CPI: 数据缺失/未采集")
            except Exception as exc:
                logger.warning("抓取 BLS CPI 失败: %s", exc)
                result["errors"].append(f"BLS CPI: {exc}")

        if self.macro_config.get("treasury_debt_enabled", True):
            try:
                items = self._fetch_treasury_debt()
                result["items"].extend(items)
                if not items:
                    result["errors"].append("U.S. Treasury Debt: 数据缺失/未采集")
            except Exception as exc:
                logger.warning("抓取美国财政部债务数据失败: %s", exc)
                result["errors"].append(f"U.S. Treasury Debt: {exc}")

        if self.macro_config.get("pboc_m2_enabled", True):
            try:
                item = self._fetch_pboc_m2()
                if item:
                    result["items"].append(item)
                else:
                    result["errors"].append("PBOC M2: 数据缺失/未采集")
            except Exception as exc:
                logger.warning("抓取中国人民银行 M2 数据失败: %s", exc)
                result["errors"].append(f"PBOC M2: {exc}")

        for asset in self.macro_config.get("market_proxies", []):
            try:
                item = self._fetch_market_proxy(asset)
                if item:
                    result["items"].append(item)
                else:
                    result["errors"].append(
                        f"{asset.get('name', asset.get('symbol', 'unknown'))}: 无有效行情"
                    )
            except Exception as exc:
                name = asset.get("name", asset.get("symbol", "unknown"))
                logger.warning("抓取宏观市场代理失败 %s: %s", name, exc)
                result["errors"].append(f"{name}: {exc}")
            time.sleep(0.2)

        logger.info("宏观指标抓取完成，共 %d 个序列", len(result["items"]))
        return result

    def _fetch_treasury_debt(self) -> List[Dict[str, Any]]:
        """抓取美国财政部官方债务规模并计算中长期趋势。"""
        history_years = max(
            5, int(self.macro_config.get("treasury_history_years", 5))
        )
        page_size = min(max(history_years * 270, 1500), 10000)
        data = self._request_json(
            self.macro_config.get("treasury_debt_url", self.TREASURY_DEBT_URL),
            {
                "fields": (
                    "record_date,tot_pub_debt_out_amt,debt_held_public_amt,"
                    "intragov_hold_amt"
                ),
                "sort": "-record_date",
                "page[size]": page_size,
            },
        )
        rows = [row for row in data.get("data", []) if row.get("record_date")]
        if not rows:
            return []
        rows.sort(key=lambda row: row["record_date"], reverse=True)
        latest = rows[0]
        latest_date = datetime.strptime(latest["record_date"], "%Y-%m-%d")
        one_year_row = self._nearest_historical_row(
            rows, self._years_before(latest_date, 1)
        )
        five_year_row = self._nearest_historical_row(
            rows, self._years_before(latest_date, 5)
        )

        series = [
            (
                "美国联邦债务总额",
                "US_TOTAL_PUBLIC_DEBT",
                "tot_pub_debt_out_amt",
                "美国财政部每日公布的联邦债务总额",
            ),
            (
                "美国公众持有联邦债务",
                "US_DEBT_HELD_PUBLIC",
                "debt_held_public_amt",
                "由政府外部投资者及美联储等持有的联邦债务",
            ),
            (
                "美国政府内部持有债务",
                "US_INTRAGOV_DEBT",
                "intragov_hold_amt",
                "美国政府信托基金等内部账户持有的联邦债务",
            ),
        ]
        items: List[Dict[str, Any]] = []
        for name, symbol, field, interpretation in series:
            latest_value = self._to_float(latest.get(field))
            if latest_value is None:
                continue
            one_year_value = self._to_float((one_year_row or {}).get(field))
            five_year_value = self._to_float((five_year_row or {}).get(field))
            items.append(
                {
                    "name": name,
                    "symbol": symbol,
                    "category": "fiscal_debt",
                    "unit": "万亿美元",
                    "source": "U.S. Treasury Fiscal Data",
                    "source_url": self.macro_config.get(
                        "treasury_debt_url", self.TREASURY_DEBT_URL
                    ),
                    "value": round(latest_value / 1_000_000_000_000, 3),
                    "raw_value_usd": latest_value,
                    "value_date": latest["record_date"],
                    "change_1y_pct": self._growth_pct(latest_value, one_year_value),
                    "cagr_5y_pct": self._cagr_pct(
                        latest_value, five_year_value, 5
                    ),
                    "interpretation": f"{interpretation}；官方日度数据，非GDP占比",
                    "is_official": True,
                    "is_proxy": False,
                }
            )
        return items

    def _fetch_pboc_m2(self) -> Optional[Dict[str, Any]]:
        """从人民银行金融统计报告抓取 M2 余额与同比趋势。"""
        report_limit = max(
            2, min(int(self.macro_config.get("pboc_report_limit", 4)), 12)
        )
        configured_urls = self.macro_config.get("pboc_m2_report_urls", [])
        report_urls = [
            {"url": url, "title": ""} for url in configured_urls if url
        ]
        if not report_urls:
            report_urls = self._discover_pboc_report_urls(report_limit)
        if not report_urls:
            return None

        observations: List[Dict[str, Any]] = []
        for report in report_urls[:report_limit]:
            try:
                html = self._request_text(report["url"])
                parsed = self._parse_pboc_m2_report(
                    html, report.get("title", ""), report["url"]
                )
                if parsed:
                    observations.append(parsed)
            except Exception as exc:
                logger.warning("解析人民银行 M2 报告失败 %s: %s", report["url"], exc)
        if not observations:
            return None

        observations.sort(key=lambda item: item["value_date"], reverse=True)
        latest = observations[0]
        previous = observations[1] if len(observations) > 1 else None
        yoy_change = None
        if previous and latest.get("yoy_pct") is not None:
            previous_yoy = previous.get("yoy_pct")
            if previous_yoy is not None:
                yoy_change = round(latest["yoy_pct"] - previous_yoy, 2)
        return {
            "name": "中国广义货币M2余额",
            "symbol": "CN_M2",
            "category": "money_supply",
            "unit": "万亿元人民币",
            "source": "中国人民银行",
            "source_url": latest["source_url"],
            "value": latest["balance_trillion_cny"],
            "raw_value_trillion_cny": latest["balance_trillion_cny"],
            "value_date": latest["value_date"],
            "yoy_pct": latest.get("yoy_pct"),
            "change_1m_pct_pts": yoy_change,
            "history": observations,
            "interpretation": (
                "人民银行官方金融统计报告；同比增速反映货币扩张速度，"
                "不等同于信用已流入实体经济或资产市场"
            ),
            "is_official": True,
            "is_proxy": False,
        }

    def _discover_pboc_report_urls(self, limit: int) -> List[Dict[str, str]]:
        index_url = self.macro_config.get(
            "pboc_m2_index_url", self.PBOC_REPORT_INDEX_URL
        )
        html = self._request_text(index_url)
        soup = BeautifulSoup(html, "html.parser")
        reports: List[Dict[str, str]] = []
        seen = set()
        for anchor in soup.find_all("a", href=True):
            title = " ".join(anchor.get_text(" ", strip=True).split())
            if "金融统计数据报告" not in title:
                continue
            url = urljoin(index_url, anchor["href"])
            if url in seen:
                continue
            seen.add(url)
            reports.append({"url": url, "title": title})
        reports.sort(
            key=lambda report: self._extract_chinese_year_month(report["title"])
            or "",
            reverse=True,
        )
        return reports[:limit]

    @classmethod
    def _parse_pboc_m2_report(
        cls, html: str, title: str, source_url: str
    ) -> Optional[Dict[str, Any]]:
        text = " ".join(BeautifulSoup(html, "html.parser").get_text(" ", strip=True).split())
        normalized = (
            text.replace("（", "(")
            .replace("）", ")")
            .replace("，", ",")
            .replace("％", "%")
        )
        pattern = re.compile(
            r"广义货币\s*\(\s*M2\s*\)\s*余额(?:为)?\s*"
            r"([\d,.]+)\s*万亿元.{0,120}?同比(?:增长|下降)\s*"
            r"([+-]?\d+(?:\.\d+)?)\s*%",
            re.IGNORECASE,
        )
        match = pattern.search(normalized)
        if not match:
            return None
        date_value = cls._extract_chinese_year_month(title or normalized[:300])
        if not date_value:
            date_value = cls._extract_chinese_year_month(normalized)
        if not date_value:
            return None
        balance = cls._to_float(match.group(1).replace(",", ""))
        yoy = cls._to_float(match.group(2))
        if balance is None or yoy is None:
            return None
        yoy_context = match.group(0)
        if "同比下降" in yoy_context and yoy > 0:
            yoy = -yoy
        return {
            "value_date": date_value,
            "balance_trillion_cny": round(balance, 3),
            "yoy_pct": round(yoy, 2),
            "source_url": source_url,
        }

    @staticmethod
    def _extract_chinese_year_month(text: str) -> Optional[str]:
        match = re.search(r"(20\d{2})年\s*(\d{1,2})月", text or "")
        if not match:
            return None
        return f"{match.group(1)}-{int(match.group(2)):02d}"

    @staticmethod
    def _years_before(value: datetime, years: int) -> datetime:
        try:
            return value.replace(year=value.year - years)
        except ValueError:
            return value.replace(year=value.year - years, day=28)

    @staticmethod
    def _nearest_historical_row(
        rows: List[Dict[str, Any]], target: datetime
    ) -> Optional[Dict[str, Any]]:
        dated_rows = []
        for row in rows:
            try:
                dated_rows.append(
                    (datetime.strptime(row["record_date"], "%Y-%m-%d"), row)
                )
            except (KeyError, TypeError, ValueError):
                continue
        if not dated_rows:
            return None
        nearest_date, nearest_row = min(
            dated_rows, key=lambda pair: abs((pair[0] - target).days)
        )
        if abs((nearest_date - target).days) > 45:
            return None
        return nearest_row

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value in {None, "", "-"}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _growth_pct(latest: float, previous: Optional[float]) -> Optional[float]:
        if previous in {None, 0}:
            return None
        return round((latest / previous - 1) * 100, 2)

    @staticmethod
    def _cagr_pct(
        latest: float, previous: Optional[float], years: int
    ) -> Optional[float]:
        if previous in {None, 0} or latest <= 0 or years <= 0:
            return None
        return round(((latest / previous) ** (1 / years) - 1) * 100, 2)

    def _fetch_bls_cpi(self) -> List[Dict[str, Any]]:
        """抓取 BLS 官方 CPI；网络或 SSL 失败时返回空列表。"""
        series_map = {
            "CUUR0000SA0": {
                "name": "美国CPI同比",
                "category": "inflation",
                "unit": "% YoY",
            },
            "CUUR0000SA0L1E": {
                "name": "美国核心CPI同比",
                "category": "inflation",
                "unit": "% YoY",
            },
        }
        end_year = datetime.now().year
        payload = {
            "seriesid": list(series_map.keys()),
            "startyear": str(end_year - 1),
            "endyear": str(end_year),
        }
        data = self._request_bls_json(payload)
        if not data or data.get("status") != "REQUEST_SUCCEEDED":
            message = (data or {}).get("message") or (data or {}).get("status") or "请求失败"
            raise RuntimeError(message)

        items: List[Dict[str, Any]] = []
        for series in data.get("Results", {}).get("series", []):
            series_id = series.get("seriesID", "")
            meta = series_map.get(series_id)
            if not meta:
                continue
            points = [
                point
                for point in series.get("data", [])
                if str(point.get("period", "")).startswith("M")
                and point.get("value") not in {None, "", "-"}
            ]
            if len(points) < 13:
                continue
            # BLS 返回通常按最新在前。
            latest = points[0]
            year_ago = points[12]
            try:
                latest_value = float(latest["value"])
                year_ago_value = float(year_ago["value"])
            except (TypeError, ValueError, KeyError):
                continue
            if year_ago_value == 0:
                continue
            yoy = round((latest_value / year_ago_value - 1) * 100, 2)
            value_date = f"{latest.get('year')}-{str(latest.get('period', 'M00')[1:]).zfill(2)}"
            items.append(
                {
                    "name": meta["name"],
                    "symbol": series_id,
                    "category": meta["category"],
                    "unit": meta["unit"],
                    "source": "BLS",
                    "value": yoy,
                    "raw_index": latest_value,
                    "value_date": value_date,
                    "change_1m_pct_pts": None,
                    "change_3m_pct_pts": None,
                    "interpretation": "官方CPI同比；月度发布，非盘中实时数据",
                    "is_official": True,
                    "is_proxy": False,
                }
            )
            if len(points) >= 2:
                try:
                    prev = float(points[1]["value"])
                    prev_year = float(points[13]["value"]) if len(points) >= 14 else None
                    if prev_year:
                        prev_yoy = (prev / prev_year - 1) * 100
                        items[-1]["change_1m_pct_pts"] = round(yoy - prev_yoy, 2)
                except (TypeError, ValueError, KeyError, IndexError):
                    pass
        return items

    def _fetch_market_proxy(self, asset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """抓取 Yahoo 市场代理，并按类型换算可解释的宏观含义。"""
        symbol = asset.get("symbol", "")
        if not symbol:
            return None
        rows = self._fetch_yahoo_rows(symbol)
        if not rows:
            return None
        closes = [row["close"] for row in rows if row.get("close") is not None]
        if not closes:
            return None

        latest = rows[-1]
        transform = asset.get("transform", "price")
        values = closes
        value = closes[-1]
        unit = asset.get("unit", "")
        interpretation = asset.get("interpretation", "")

        if transform == "implied_rate_from_100":
            values = [round(100 - close, 4) for close in closes]
            value = values[-1]
            unit = asset.get("unit", "%")
            interpretation = (
                asset.get("interpretation")
                or "联邦基金期货隐含利率=100-期货报价；反映市场对政策利率路径的定价，"
                "不是 CME FedWatch 会议加息概率"
            )

        return {
            "name": asset.get("name", symbol),
            "symbol": symbol,
            "category": asset.get("category", "macro"),
            "unit": unit,
            "source": "Yahoo Finance",
            "value": round(value, 4),
            "raw_price": closes[-1],
            "value_date": latest.get("date", ""),
            "change_1d_pct": self._pct_change(values, 1),
            "change_5d_pct": self._pct_change(values, 5),
            "change_20d_pct": self._pct_change(values, 20),
            "change_1d_pts": self._point_change(values, 1),
            "change_5d_pts": self._point_change(values, 5),
            "change_20d_pts": self._point_change(values, 20),
            "interpretation": interpretation,
            "is_official": False,
            "is_proxy": True,
        }

    def _fetch_yahoo_rows(self, symbol: str) -> List[Dict[str, Any]]:
        today = datetime.now(timezone.utc)
        start = today - timedelta(days=max(self.historical_days * 2, 120))
        data = self._request_json(
            f"{self.YAHOO_CHART_URL}/{quote(symbol, safe='')}",
            {
                "period1": int(start.timestamp()),
                "period2": int(today.timestamp()),
                "interval": "1d",
                "events": "history",
            },
        )
        result = ((data.get("chart", {}).get("result") or [None])[0]) if data else None
        if not result:
            return []
        quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
        rows = []
        for index, timestamp in enumerate(result.get("timestamp") or []):
            close = self._value_at(quote_data.get("close", []), index)
            if close is None:
                continue
            rows.append(
                {
                    "date": datetime.fromtimestamp(timestamp, timezone.utc).strftime(
                        "%Y-%m-%d"
                    ),
                    "close": close,
                }
            )
        return rows[-self.historical_days :]

    def _request_bls_json(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.post(
                    self.BLS_API_URL, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.retry_attempts - 1:
                    raise RuntimeError(f"BLS 请求失败: {exc}") from exc
                time.sleep(self.retry_delay)
        return None

    def _request_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429:
                    raise RuntimeError("HTTP 429，本轮停止该宏观数据源请求")
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.retry_attempts - 1:
                    raise RuntimeError(f"请求失败: {exc}") from exc
                time.sleep(self.retry_delay)
        return {}

    def _request_text(self, url: str) -> str:
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 429:
                    raise RuntimeError("HTTP 429，本轮停止该宏观数据源请求")
                response.raise_for_status()
                if not response.encoding or response.encoding.lower() == "iso-8859-1":
                    response.encoding = response.apparent_encoding
                return response.text
            except requests.RequestException as exc:
                if attempt == self.retry_attempts - 1:
                    raise RuntimeError(f"页面请求失败: {exc}") from exc
                time.sleep(self.retry_delay)
        return ""

    @staticmethod
    def _value_at(values: List[Any], index: int) -> Optional[float]:
        if index >= len(values):
            return None
        try:
            value = values[index]
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pct_change(values: List[float], periods: int) -> Optional[float]:
        if len(values) <= periods or values[-periods - 1] == 0:
            return None
        return round((values[-1] / values[-periods - 1] - 1) * 100, 2)

    @staticmethod
    def _point_change(values: List[float], periods: int) -> Optional[float]:
        if len(values) <= periods:
            return None
        return round(values[-1] - values[-periods - 1], 2)
