# 技术分析指标参考

## Key Metrics for A-Share Recommendations

### Volume & Turnover
| Metric | Low | Normal | High (Warning) |
|--------|-----|--------|----------------|
| 换手率 (Turnover) | <2% | 2-5% | >7% |
| 量比 (Volume Ratio) | <0.5 | 0.5-1.5 | >2.5 |

High turnover + falling price = distribution (机构派发).
Low turnover at support = potential bottom.

### Price Action Patterns
- **高开低走** (open high, close low) = bearish intraday
- **低开高走** (open low, close high) = bullish intraday
- **长上影线** (long upper shadow) = rejection at high
- **长下影线** (long lower shadow) = support found

### Support/Resistance Levels
| Level | How to Calculate |
|-------|-----------------|
| Pivot | (昨收 + 今开) / 2 |
| 压力1 (R1) | 今开 + (最高 - 最低) × 0.382 |
| 压力2 (R2) | 涨停价 (昨日收盘 × 1.2 for 科创板) |
| 支撑1 (S1) | 今日最低价 or 整数关口 |
| 支撑2 (S2) | 跌停价 (昨日收盘 × 0.8 for 科创板) |
| 止损位 | 支撑1下方 2-3% |

### Market Context
- **大盘联动**: When 上证/深证 drops >2%, individual stock analysis is secondary — systemic risk dominates
- **板块效应**: Check if the stock's sector (半导体, 新能源, etc.) is leading or lagging
- **科创板 volatility**: ±20% daily limits, much higher risk than main board

### Key Ratios
| Ratio | What It Tells You |
|-------|-------------------|
| PE > 100 | Highly speculative / growth story |
| PE > 1000 | No real earnings support — pure momentum |
| 流通市值/总市值 < 30% | Very limited free float |
| 换手率/振幅 ratio | Efficiency of price discovery |

### Daily Strategy Decision Matrix

| Scenario | Action |
|----------|--------|
| 高开 + 放量 > 昨量1.5x + 站稳压力位 | Momentum buy (risky) |
| 高开 + 缩量 | Trap — avoid |
| 平开 + 缩量震荡 | Wait |
| 低开 + 放量下跌 | Sell / don't buy |
| 低开 + 缩量企稳 | Potential buy signal |
| 跌破昨日最低价 | Stop loss trigger |
| 站稳昨日最高价 | Breakout confirmation |

### Disclaimer Required
Always include a prominent disclaimer in Chinese:
> **⚠️ 重要声明：以上建议仅供参考，不构成投资建议。股市有风险，投资需谨慎！**

And specify the data timestamp: the recommendation is based on specific closing data and does not account for after-hours events.
