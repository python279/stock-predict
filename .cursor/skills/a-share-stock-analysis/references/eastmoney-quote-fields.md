# 东方财富行情页字段映射

From `https://quote.eastmoney.com/sh<CODE>.html` or `https://quote.eastmoney.com/kcb/<CODE>.html` (科创板)

## Key Elements in Snapshot

| AX Text / Label | Meaning | Usage |
|-----------------|---------|-------|
| "最新：" + value | Current price | Primary price |
| "最高：" + value | Day high | Resistance level |
| "今开：" + value | Open | Gap direction |
| "换手率：" + value% | Turnover rate | >7% = active/distribution |
| "总市值：" + value亿 | Total market cap | Size (亿 = ×10^8) |
| "涨停价：" + value | Next day limit up | Ceiling |
| "成交量：" + value万 | Volume in 10k lots | Volume proxy |
| 标题或顶栏股价 | ±value ±value% | Daily change |
| 标题"sector" | Industry sector | Context |
| "成交额：16.96亿" | Turnover in 元 | Value traded |
| "流通市值：203亿" | Float market cap | Free-floating shares value |
| "市盈(动)：2692.69" | PE TTM | Earnings multiple |
| "振幅：4.88%" | Daily amplitude | Volatility measure |

## 股吧新闻阅读 (guba.eastmoney.com/list,<CODE>.html)

| Tab name in snapshot | Action |
|----------------------|--------|
| "资讯" | Company news |
| "公告" | Official announcements |
| "研报" | Research reports |
| "热门" | Popular sentiment posts |

## Board URL Mapping

| Board | 东方财富 URL |
|-------|-------------|
| 沪市主板 (600-603) | `https://quote.eastmoney.com/sh<CODE>.html` |
| 深市主板 (000-002) | `https://quote.eastmoney.com/sz<CODE>.html` |
| 创业板 (300) | `https://quote.eastmoney.com/sz<CODE>.html` |
| 科创板 (688) | `https://quote.eastmoney.com/kcb/<CODE>.html` |
| 北交所 | Not available on eastmoney quote |
