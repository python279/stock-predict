"""市场行情抓取模块，提供美股 ETF 与 A 股指数/ETF 的统一趋势数据。"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """标准化的市场行情快照。"""

    name: str
    symbol: str
    market: str
    sector: str
    source: str
    price: Optional[float]
    price_date: str
    change_1d_pct: Optional[float]
    change_5d_pct: Optional[float]
    change_20d_pct: Optional[float]
    ma5: Optional[float]
    ma20: Optional[float]
    ma60: Optional[float]
    volume: Optional[float]
    volume_ratio_20d: Optional[float]
    drawdown_20d_pct: Optional[float]
    volatility_20d_pct: Optional[float]
    risk_signals: List[str]
    risk_level: str
    trend_signal: str
    investment_targets: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为供存储和 LLM 使用的字典。"""
        return self.__dict__.copy()


class MarketDataFetcher:
    """抓取配置化的美股和 A 股宽基、行业指数或 ETF。"""

    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
    TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.market_config = config.get("markets", {})
        fetcher_config = config.get("fetcher", {})
        self.enabled = self.market_config.get("enabled", False)
        self.timeout = self.market_config.get(
            "timeout_seconds", fetcher_config.get("timeout_seconds", 30)
        )
        self.retry_attempts = self.market_config.get(
            "retry_attempts", fetcher_config.get("retry_attempts", 3)
        )
        self.retry_delay = self.market_config.get(
            "retry_delay_seconds", fetcher_config.get("retry_delay_seconds", 5)
        )
        self.historical_days = self.market_config.get("historical_days", 90)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": fetcher_config.get("user_agent", "NewsAnalyzer/1.0")}
        )

    def fetch_all_markets(self) -> Dict[str, Any]:
        """抓取所有配置资产，单个资产失败不会中断日报。"""
        result: Dict[str, Any] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
            "errors": [],
        }
        if not self.enabled:
            logger.info("市场行情抓取未启用")
            return result

        for asset in self.market_config.get("assets", []):
            try:
                item = self._fetch_asset(asset)
                if item:
                    result["items"].append(item.to_dict())
                else:
                    result["errors"].append(
                        f"{asset.get('name', asset.get('symbol', 'unknown'))}: 无有效行情"
                    )
            except Exception as exc:  # 外部数据源必须降级
                name = asset.get("name", asset.get("symbol", "unknown"))
                logger.warning("抓取市场行情失败 %s: %s", name, exc)
                result["errors"].append(f"{name}: {exc}")
            time.sleep(0.2)

        logger.info("市场行情抓取完成，共 %d 个资产", len(result["items"]))
        self._add_sector_risk_assessments(result["items"])
        return result

    def _fetch_asset(self, asset: Dict[str, Any]) -> Optional[MarketData]:
        provider = asset.get("provider", "").lower()
        market = asset.get("market", "").upper()
        if provider == "yahoo" or market == "US":
            rows = self._fetch_yahoo_rows(asset.get("symbol", ""))
            source = "Yahoo Finance"
        elif provider == "eastmoney" or market == "CN":
            rows = self._fetch_a_share_rows(asset)
            source = "东方财富 / 腾讯财经"
        else:
            raise ValueError(f"不支持的市场数据提供商: {provider or market}")
        return self._build_market_data(asset, rows, source)

    def _fetch_yahoo_rows(self, symbol: str) -> List[Dict[str, Any]]:
        if not symbol:
            return []
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
            if close is not None:
                rows.append(
                    {
                        "date": datetime.fromtimestamp(
                            timestamp, timezone.utc
                        ).strftime("%Y-%m-%d"),
                        "close": close,
                        "volume": self._value_at(quote_data.get("volume", []), index),
                    }
                )
        return rows[-self.historical_days:]

    def _fetch_a_share_rows(self, asset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """以腾讯前复权日线为历史基线，以东财快照补充交易日最新价。"""
        tencent_symbol = asset.get("tencent_symbol", "")
        secid = asset.get("secid", "")
        if not tencent_symbol or not secid:
            raise ValueError("A 股资产必须配置 tencent_symbol 和 secid")
        data = self._request_json(
            self.TENCENT_KLINE_URL,
            {"param": f"{tencent_symbol},day,,,{self.historical_days},qfq"},
        )
        days = ((data.get("data", {}).get(tencent_symbol, {}) or {}).get("day") or [])
        rows = [
            {
                "date": str(day[0]),
                "close": self._parse_float(day[2]),
                "volume": self._parse_float(day[5]) if len(day) > 5 else None,
            }
            for day in days
            if len(day) > 2 and self._parse_float(day[2]) is not None
        ]

        try:
            quote_data = self._request_json(
                self.EASTMONEY_QUOTE_URL,
                {
                    "secid": secid,
                    "fields": "f43,f47,f55,f57,f58,f60",
                },
            ).get("data") or {}
        except RuntimeError as exc:
            # 东方财富快照不可用时，腾讯日线仍可提供上一个交易日的趋势基线。
            logger.warning("东财快照不可用，使用腾讯历史日线 %s: %s", secid, exc)
            quote_data = {}
        latest_price = self._parse_float(quote_data.get("f43"))
        if latest_price is not None:
            latest_price /= 100
            today = datetime.now().strftime("%Y-%m-%d")
            latest_volume = self._parse_float(quote_data.get("f47"))
            if rows and rows[-1]["date"] == today:
                rows[-1].update({"close": latest_price, "volume": latest_volume})
            elif not rows or latest_price != rows[-1]["close"]:
                rows.append({"date": today, "close": latest_price, "volume": latest_volume})
        return rows[-self.historical_days:]

    def _build_market_data(
        self, asset: Dict[str, Any], rows: List[Dict[str, Any]], source: str
    ) -> Optional[MarketData]:
        if not rows:
            return None
        closes = [row["close"] for row in rows if row.get("close") is not None]
        if not closes:
            return None
        volumes = [row["volume"] for row in rows if row.get("volume") is not None]
        latest = rows[-1]
        latest_volume = self._parse_float(latest.get("volume"))
        volume_ratio = None
        if latest_volume is not None and len(volumes) >= 20:
            average_volume = sum(volumes[-20:]) / 20
            if average_volume:
                volume_ratio = round(latest_volume / average_volume, 2)

        return MarketData(
            name=asset.get("name", asset.get("symbol", "")),
            symbol=asset.get("symbol", asset.get("secid", "")),
            market=asset.get("market", ""),
            sector=asset.get("sector", "broad"),
            source=source,
            price=closes[-1],
            price_date=latest.get("date", ""),
            change_1d_pct=self._pct_change(closes, 1),
            change_5d_pct=self._pct_change(closes, 5),
            change_20d_pct=self._pct_change(closes, 20),
            ma5=self._moving_average(closes, 5),
            ma20=self._moving_average(closes, 20),
            ma60=self._moving_average(closes, 60),
            volume=latest_volume,
            volume_ratio_20d=volume_ratio,
            drawdown_20d_pct=self._drawdown(closes, 20),
            volatility_20d_pct=self._volatility(closes, 20),
            risk_signals=[],
            risk_level="待评估",
            trend_signal=self._trend_signal(closes),
            investment_targets=asset.get("investment_targets", {}),
        )

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
        return MarketDataFetcher._parse_float(values[index]) if index < len(values) else None

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        try:
            if value is None or str(value).strip().lower() in {"", "n/a", "nan"}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _moving_average(values: List[float], window: int) -> Optional[float]:
        if len(values) < window:
            return None
        return round(sum(values[-window:]) / window, 4)

    @staticmethod
    def _pct_change(values: List[float], periods: int) -> Optional[float]:
        if len(values) <= periods or values[-periods - 1] == 0:
            return None
        return round((values[-1] / values[-periods - 1] - 1) * 100, 2)

    def _trend_signal(self, closes: List[float]) -> str:
        ma5 = self._moving_average(closes, 5)
        ma20 = self._moving_average(closes, 20)
        ma60 = self._moving_average(closes, 60)
        change_20d = self._pct_change(closes, 20)
        if ma20 is None:
            return "数据不足"
        if closes[-1] > ma20 and (ma60 is None or ma20 > ma60) and (change_20d or 0) > 3:
            return "上升趋势"
        if closes[-1] < ma20 and (ma60 is None or ma20 < ma60) and (change_20d or 0) < -3:
            return "下降趋势"
        if ma5 is not None and closes[-1] > ma5 > ma20:
            return "短期转强"
        if ma5 is not None and closes[-1] < ma5 < ma20:
            return "短期转弱"
        return "震荡"

    @staticmethod
    def _drawdown(closes: List[float], window: int) -> Optional[float]:
        """计算现价相对窗口内高点的回撤比例。"""
        if len(closes) < window:
            return None
        peak = max(closes[-window:])
        if peak == 0:
            return None
        return round((closes[-1] / peak - 1) * 100, 2)

    @staticmethod
    def _volatility(closes: List[float], window: int) -> Optional[float]:
        """计算日收益率的20日滚动波动率，供风险分层而非收益预测使用。"""
        if len(closes) < window + 1:
            return None
        returns = [
            closes[index] / closes[index - 1] - 1
            for index in range(-window + 1, 0)
            if closes[index - 1] != 0
        ]
        if len(returns) < 2:
            return None
        average = sum(returns) / len(returns)
        variance = sum((item - average) ** 2 for item in returns) / (len(returns) - 1)
        return round((variance ** 0.5) * 100, 2)

    @staticmethod
    def _add_sector_risk_assessments(items: List[Dict[str, Any]]) -> None:
        """标注板块急跌脆弱性，不将它包装成黑天鹅的确定性预测。"""
        broad_by_market = {
            item.get("market"): item
            for item in items
            if item.get("sector") == "broad"
        }
        for item in items:
            if item.get("sector") not in {"tech", "finance", "consumer"}:
                continue

            signals: List[str] = []
            broad = broad_by_market.get(item.get("market"))
            change_5d = item.get("change_5d_pct")
            if (
                broad
                and change_5d is not None
                and broad.get("change_5d_pct") is not None
                and change_5d <= broad["change_5d_pct"] - 3
            ):
                signals.append("5日相对宽基显著跑输")
            if (item.get("change_20d_pct") or 0) <= -8:
                signals.append("20日跌幅超过8%")
            if (item.get("drawdown_20d_pct") or 0) <= -8:
                signals.append("距20日高点回撤超过8%")
            if (
                item.get("price") is not None
                and item.get("ma20") is not None
                and item.get("ma60") is not None
                and item["price"] < item["ma20"] < item["ma60"]
            ):
                signals.append("价格跌破MA20且MA20低于MA60")
            if (
                (item.get("volume_ratio_20d") or 0) >= 1.3
                and (item.get("change_1d_pct") or 0) < 0
            ):
                signals.append("下跌日放量")

            item["risk_signals"] = signals or ["未触发量价脆弱性阈值"]
            item["risk_level"] = (
                "高" if len(signals) >= 3 else "中" if len(signals) >= 2
                else "低" if len(signals) == 1 else "观察"
            )
