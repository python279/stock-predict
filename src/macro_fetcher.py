"""宏观指标抓取：美国通胀与美联储利率预期。

官方 CPI 优先尝试 BLS 公开接口；失败时降级为可验证的市场代理
（TIPS ETF、联邦基金期货隐含利率、短端/长端国债收益率），并在输出中
明确标注“数据缺失/市场代理”，不得伪装成官方打印。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


class MacroFetcher:
    """抓取通胀与利率政策相关宏观输入。"""

    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

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
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.retry_attempts - 1:
                    raise RuntimeError(f"请求失败: {exc}") from exc
                time.sleep(self.retry_delay)
        return {}

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
