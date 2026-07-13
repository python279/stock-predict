"""
大宗商品价格抓取模块
使用免费 CSV 行情源抓取主要大宗商品价格和趋势指标。
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class CommodityData:
    """大宗商品行情和趋势数据"""

    name: str
    symbol: str
    category: str
    unit: str
    source: str
    price: Optional[float]
    price_date: str
    change_1d_pct: Optional[float]
    change_5d_pct: Optional[float]
    change_20d_pct: Optional[float]
    ma5: Optional[float]
    ma20: Optional[float]
    ma60: Optional[float]
    trend_signal: str
    investment_targets: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于存储和传给 LLM。"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "category": self.category,
            "unit": self.unit,
            "source": self.source,
            "price": self.price,
            "price_date": self.price_date,
            "change_1d_pct": self.change_1d_pct,
            "change_5d_pct": self.change_5d_pct,
            "change_20d_pct": self.change_20d_pct,
            "ma5": self.ma5,
            "ma20": self.ma20,
            "ma60": self.ma60,
            "trend_signal": self.trend_signal,
            "investment_targets": self.investment_targets,
        }


class CommodityFetcher:
    """大宗商品价格抓取器"""

    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, config: Dict[str, Any]):
        """
        初始化大宗商品抓取器。

        Args:
            config: 配置字典
        """
        self.config = config
        self.commodity_config = config.get("commodities", {})
        fetcher_config = config.get("fetcher", {})

        self.enabled = self.commodity_config.get("enabled", False)
        self.timeout = self.commodity_config.get(
            "timeout_seconds",
            fetcher_config.get("timeout_seconds", 30),
        )
        self.retry_attempts = self.commodity_config.get(
            "retry_attempts",
            fetcher_config.get("retry_attempts", 3),
        )
        self.retry_delay = self.commodity_config.get(
            "retry_delay_seconds",
            fetcher_config.get("retry_delay_seconds", 5),
        )
        self.historical_days = self.commodity_config.get("historical_days", 90)
        self.source = self.commodity_config.get("provider", "yahoo")

        user_agent = fetcher_config.get("user_agent", "NewsAnalyzer/1.0")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch_all_commodities(self) -> Dict[str, Any]:
        """
        抓取全部配置的大宗商品数据。

        Returns:
            包含抓取时间、数据列表和错误列表的字典
        """
        result = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "items": [],
            "errors": [],
        }

        if not self.enabled:
            logger.info("大宗商品抓取未启用")
            return result

        assets = self.commodity_config.get("assets", [])
        if not assets:
            logger.warning("未配置大宗商品资产列表")
            return result

        for asset in assets:
            try:
                commodity = self._fetch_yahoo_asset(asset)
                if commodity:
                    result["items"].append(commodity.to_dict())
                    logger.info(
                        f"抓取大宗商品成功: {commodity.name} "
                        f"{commodity.price} {commodity.unit}"
                    )
                time.sleep(0.2)
            except Exception as e:
                name = asset.get("name", asset.get("symbol", "unknown"))
                error_msg = f"{name}: {e}"
                result["errors"].append(error_msg)
                logger.warning(f"抓取大宗商品失败 {error_msg}")

        logger.info(f"大宗商品抓取完成，共 {len(result['items'])} 个品种")
        return result

    def _fetch_yahoo_asset(self, asset: Dict[str, Any]) -> Optional[CommodityData]:
        """从 Yahoo Finance 抓取单个品种的日线数据。"""
        symbol = asset.get("symbol", "")
        if not symbol:
            return None

        rows = self._fetch_daily_rows(symbol)
        if not rows:
            return None

        closes = [row["close"] for row in rows if row.get("close") is not None]
        if not closes:
            return None

        latest = rows[-1]
        price = latest.get("close")

        return CommodityData(
            name=asset.get("name", symbol),
            symbol=symbol,
            category=asset.get("category", "commodity"),
            unit=asset.get("unit", ""),
            source="Yahoo Finance",
            price=price,
            price_date=latest.get("date", ""),
            change_1d_pct=self._pct_change(closes, 1),
            change_5d_pct=self._pct_change(closes, 5),
            change_20d_pct=self._pct_change(closes, 20),
            ma5=self._moving_average(closes, 5),
            ma20=self._moving_average(closes, 20),
            ma60=self._moving_average(closes, 60),
            trend_signal=self._trend_signal(closes),
            investment_targets=asset.get("investment_targets", {}),
        )

    def _fetch_daily_rows(self, symbol: str) -> List[Dict[str, Any]]:
        """抓取日线 JSON 并解析为价格行。"""
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=max(self.historical_days * 2, 120))
        period1 = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        period2 = int(datetime.now(timezone.utc).timestamp())
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
        }

        data = self._make_json_request(f"{self.YAHOO_CHART_URL}/{symbol}", params=params)
        if not data:
            return []

        chart = data.get("chart", {})
        if chart.get("error"):
            logger.warning(f"Yahoo Finance 返回错误 {symbol}: {chart.get('error')}")
            return []

        results = chart.get("result") or []
        if not results:
            return []

        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]

        rows: List[Dict[str, Any]] = []
        for index, timestamp in enumerate(timestamps):
            close = self._value_at(quote.get("close", []), index)
            if close is None:
                continue
            price_date = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")
            rows.append(
                {
                    "date": price_date,
                    "open": self._value_at(quote.get("open", []), index),
                    "high": self._value_at(quote.get("high", []), index),
                    "low": self._value_at(quote.get("low", []), index),
                    "close": close,
                    "volume": self._value_at(quote.get("volume", []), index),
                }
            )

        return rows[-self.historical_days:]

    def _make_json_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """发起 JSON 请求（带重试机制）。"""
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"大宗商品请求失败 (尝试 {attempt + 1}/{self.retry_attempts}): {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"大宗商品请求最终失败: {url}")

        return {}

    def _value_at(self, values: List[Any], index: int) -> Optional[float]:
        """安全读取列表中的数值。"""
        if index >= len(values):
            return None
        return self._parse_float(values[index])

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        """安全解析浮点数。"""
        try:
            if value is None or str(value).strip().lower() in {"", "n/a", "nan"}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _moving_average(values: List[float], window: int) -> Optional[float]:
        """计算移动平均。"""
        if len(values) < window:
            return None
        return round(sum(values[-window:]) / window, 4)

    @staticmethod
    def _pct_change(values: List[float], periods: int) -> Optional[float]:
        """计算指定周期涨跌幅。"""
        if len(values) <= periods:
            return None
        base = values[-periods - 1]
        latest = values[-1]
        if base == 0:
            return None
        return round((latest - base) / base * 100, 2)

    def _trend_signal(self, closes: List[float]) -> str:
        """根据均线和短期涨跌幅生成趋势信号。"""
        if not closes:
            return "数据不足"

        latest = closes[-1]
        ma5 = self._moving_average(closes, 5)
        ma20 = self._moving_average(closes, 20)
        ma60 = self._moving_average(closes, 60)
        change_20d = self._pct_change(closes, 20)

        if ma20 is None:
            return "数据不足"

        if latest > ma20 and (ma60 is None or ma20 > ma60) and (change_20d or 0) > 3:
            return "上升趋势"
        if latest < ma20 and (ma60 is None or ma20 < ma60) and (change_20d or 0) < -3:
            return "下降趋势"
        if ma5 is not None and latest > ma5 > ma20:
            return "短期转强"
        if ma5 is not None and latest < ma5 < ma20:
            return "短期转弱"
        return "震荡"
