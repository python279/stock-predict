"""黄金脆弱性信号：价格过热 × 利率/美元逆风背离。

只输出情景预警等级和触发依据，不是价格预测；任一输入缺失时写明未采集。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# (symbol, 名称, 变化字段, 单位后缀, 逆风方向: +1 表示上升利空黄金)
DRIVERS = (
    ("DFII10", "10年期实际利率", "change_20d_pts", "pt", 1, "real_yield_20d_pts"),
    ("^TNX", "10年期名义利率", "change_20d_pts", "pt", 1, "nominal_yield_20d_pts"),
    ("ZQ=F", "联邦基金期货隐含利率", "change_20d_pts", "pt", 1, "policy_rate_20d_pts"),
    ("DX-Y.NYB", "美元指数", "change_20d_pct", "%", 1, "dxy_20d_pct"),
    ("TIP", "通胀保值债ETF", "change_20d_pct", "%", -1, "tip_20d_pct"),
)

DEFAULT_THRESHOLDS = {
    "real_yield_20d_pts": 0.15,
    "nominal_yield_20d_pts": 0.20,
    "policy_rate_20d_pts": 0.15,
    "dxy_20d_pct": 2.0,
    "tip_20d_pct": 1.0,
    "elevated_ma60_deviation_pct": 5.0,
    "overheat_change_20d_pct": 10.0,
    "overheat_ma60_deviation_pct": 8.0,
}


def evaluate_gold_signals(
    commodity_data: Optional[Dict[str, Any]],
    macro_data: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or {}
    thresholds = {**DEFAULT_THRESHOLDS, **config.get("thresholds", {})}
    gold_symbol = config.get("gold_symbol", "GC=F")

    gold = next(
        (
            item
            for item in (commodity_data or {}).get("items", [])
            if item.get("symbol") == gold_symbol
        ),
        None,
    )
    macro_items = {
        item.get("symbol"): item for item in (macro_data or {}).get("items", [])
    }

    drivers: List[Dict[str, Any]] = []
    headwinds: List[str] = []
    tailwinds: List[str] = []
    missing: List[str] = []
    for symbol, name, field, suffix, direction, threshold_key in DRIVERS:
        item = macro_items.get(symbol)
        change = item.get(field) if item else None
        if change is None:
            missing.append(name)
            continue
        threshold = float(thresholds[threshold_key])
        signed = float(change) * direction
        if signed >= threshold:
            stance = "逆风"
            headwinds.append(f"{name}20日{float(change):+.2f}{suffix}")
        elif signed <= -threshold:
            stance = "顺风"
            tailwinds.append(f"{name}20日{float(change):+.2f}{suffix}")
        else:
            stance = "中性"
        drivers.append(
            {
                "name": name,
                "symbol": symbol,
                "value": item.get("value"),
                "value_date": item.get("value_date"),
                "change_5d": item.get(field.replace("20d", "5d")),
                "change_20d": change,
                "unit": suffix,
                "stance": stance,
            }
        )

    result: Dict[str, Any] = {
        "gold_symbol": gold_symbol,
        "gold": None,
        "drivers": drivers,
        "headwinds": headwinds,
        "tailwinds": tailwinds,
        "missing": missing,
        "level": "数据缺失",
        "reason": "黄金行情未采集",
    }
    if not gold:
        return result

    deviation = gold.get("ma60_deviation_pct")
    if deviation is None and gold.get("price") and gold.get("ma60"):
        deviation = round((gold["price"] / gold["ma60"] - 1) * 100, 2)
    overheat = gold.get("overheat_signal") or "数据不足"
    change_20d = gold.get("change_20d_pct")
    if overheat == "数据不足" and change_20d is not None and deviation is not None:
        if (
            change_20d >= float(thresholds["overheat_change_20d_pct"])
            and deviation >= float(thresholds["overheat_ma60_deviation_pct"])
        ):
            overheat = "过热"
    elevated = overheat in {"过热", "偏热"} or (
        deviation is not None
        and deviation >= float(thresholds["elevated_ma60_deviation_pct"])
    )
    result["gold"] = {
        "price": gold.get("price"),
        "price_date": gold.get("price_date"),
        "change_20d_pct": gold.get("change_20d_pct"),
        "ma60_deviation_pct": deviation,
        "rsi14": gold.get("rsi14"),
        "trend_signal": gold.get("trend_signal"),
        "overheat_signal": overheat,
        "elevated": elevated,
    }

    if elevated and len(headwinds) >= 2:
        level, reason = "高", "金价处于高位且利率/美元多项转为逆风，出现高位背离"
    elif elevated and headwinds:
        level, reason = "中", "金价处于高位且已有单项利率/美元逆风"
    elif overheat == "过热":
        level, reason = "中", "金价动量与均线偏离过热，驱动面尚未确认逆风"
    elif len(headwinds) >= 2:
        level, reason = "中", "利率/美元多项转为逆风，金价尚未过热"
    else:
        level, reason = "低", "未见过热与驱动逆风共振"
    if missing:
        reason += f"；未采集：{'、'.join(missing)}"
    result["level"] = level
    result["reason"] = reason
    return result
