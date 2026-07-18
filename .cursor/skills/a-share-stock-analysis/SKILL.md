---
name: a-share-stock-analysis
description: "Research, analyze, and provide daily trading recommendations for Chinese A-share stocks (中国A股). Covers real-time quotes, technical analysis (support/resistance/volume), sentiment mining, and structured buy/sell advice."
version: 1.0.0
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [a-share, stock, analysis, trading, 股票, 财经, 技术分析, 东方财富, 股市]
    related_skills: [google-workspace, notion]
---

# A-Share Stock Analysis (中国A股分析)

## When to Use

Use this skill when the user asks for stock trading advice on Chinese A-shares, including:
- "分析XX股票"
- "XX的买入卖出建议"
- "每天给XX的交易建议"
- "结合趋势和舆论分析XX股票"
- "XX股现在该买还是该卖"

## Data Sources (Reliable) — Prefer JSON APIs over HTML scraping

### 🥇 Primary: 东方财富 Push API (JSON, no CAPTCHA)
The most reliable real-time data source. Returns structured JSON, works from `curl` in terminal, no browser needed.

| Endpoint | Purpose |
|----------|---------|
| `https://push2.eastmoney.com/api/qt/stock/get?secid=1.{CODE}&fields={FIELDS}` | Real-time quote snapshot |
| `https://push2.eastmoney.com/api/qt/stock/gett?secid=1.{CODE}&fields={FIELDS}` | Real-time tick (same fields, lighter response) |

**Key field codes** (values are in 分/cents — divide by 100):
| Field | Meaning | ÷100? |
|-------|---------|-------|
| f43 | 最新价 (current price) | Yes |
| f44 | 今开 (open) | Yes |
| f45 | 最高 (high) | Yes |
| f46 | 最低 (low) | Yes |
| f47 | 成交量 (volume, 手) | No |
| f48 | 成交额 (turnover) | No (raw) |
| f49 | 换手率 (turnover rate) | Yes |
| f50 | 振幅 (amplitude) | Yes |
| f51 | 涨停价 (limit up) | Yes |
| f52 | 跌停价 (limit down) | Yes |
| f55 | 涨跌幅 (change %) | No (decimal) |
| f57 | 股票代码 | No |
| f58 | 股票名称 | No |
| f60 | 昨收 (prev close) | Yes |
| f116 | 总市值 (total mkt cap) | No |
| f117 | 流通市值 (float mkt cap) | No |
| f162 | 市盈率(动) (PE TTM) | Yes |
| f167 | 上涨家数 | No |
| f168 | 下跌家数 | No |
| f170 | 平盘家数 | No |
| f171 | 总家数 | No |
| f292 | 交易状态 (1=盘前,2=交易中,3=收盘) | No |

**Usage** (from terminal):
```bash
# All key fields
curl -s "https://push2.eastmoney.com/api/qt/stock/get?secid=1.688795&fields=f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162,f167,f168,f169,f170,f171" -H "User-Agent: Mozilla/5.0"
```

See `references/eastmoney-push-api-fields.md` for the complete field reference.

### 🥈 Secondary: 腾讯金融API (历史K线)
| Source | URL Pattern | Use Case |
|--------|-------------|----------|
| **腾讯金融API** | `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh{CODE},day,,,120,qfq` | Historical daily K-line data. Reliable, no CAPTCHA. Returns JSON directly. Each entry: `[date, open, close, high, low, volume]`. |

**Usage** (from terminal):
```bash
curl -s "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh688795,day,,,120,qfq" -H "User-Agent: Mozilla/5.0"
```

### 🥉 Fallback: HTML Pages (when API unavailable)
| Source | URL Pattern | Use Case |
|--------|-------------|----------|
| **东方财富行情** | `https://quote.eastmoney.com/sh{CODE}.html` (沪/科创板) | Manual verification |
| **东方财富行情** | `https://quote.eastmoney.com/sz{CODE}.html` (深/创业板) | Manual verification |
| **东方财富股吧** | `https://guba.eastmoney.com/list,{CODE}.html` | Sentiment mining |
| **新浪财经** | `https://finance.sina.com.cn/realstock/company/sh{CODE}/nc.shtml` | Alternative quote |

### ⚠️ Pitfall: Search Engines Blocked
Google and Baidu frequently trigger **CAPTCHA/验证码** when accessed programmatically. **Do NOT use them as primary data sources.** Always go directly to 东方财富 APIs.

## 💡 Finding a Stock's Code

**Preferred method** — use the 东方财富 suggest API (no browser needed):
```bash
curl -s "https://searchadapter.eastmoney.com/api/suggest/get?input={URL_ENCODED_NAME}&type=14&token=D43BF722C8E33BDC906FB84D85E326E8" -H "User-Agent: Mozilla/5.0"
```
Returns JSON with `Code`, `Name`, `SecurityTypeName` (板块). Example:
```json
{"QuotationCodeTable":{"Data":[{"Code":"688795","Name":"摩尔线程-U","SecurityTypeName":"科创板"}]}}
```

**Fallback** — browser-based search:
1. Navigate to `https://www.eastmoney.com/`
2. Type the stock name into the search box
3. Click "查行情" (search quote)
4. Look under "相关证券" for the exact stock with code and board
5. Note the board prefix:
   - **SH** (沪市主板): `sh` — e.g., sh600000
   - **SZ** (深市/创业板): `sz` — e.g., sz300750
   - **科创板 (STAR Market)**: `sh` — e.g., sh688795
6. Navigate to `https://quote.eastmoney.com/sh{CODE}.html` or `https://quote.eastmoney.com/sz{CODE}.html`

## Data to Collect (from 东方财富行情页)

Key fields to extract:
- **最新价** (current price) + **涨跌幅** (change %)
- **今开/最高/最低** (open/high/low)
- **昨收** (previous close)
- **成交量/成交额** (volume/turnover)
- **换手率** (turnover rate) — high (>5%) = active trading
- **振幅** (amplitude)
- **总市值/流通市值** (total/float market cap)
- **市盈率(动)** (PE TTM) — very high = speculative
- **涨停价/跌停价** (limit up/down prices for next day)
- **委比/量比** (order ratios)

## Sentiment Mining (from 股吧)

Navigate to `https://guba.eastmoney.com/list,<CODE>.html`

Key things to look for:
- Recent **news/announcements** (look for "资讯" or "公告" tab entries)
- **Hot posts** by reading counts (high 阅读 = attention)
- **Company events** (子司增资, 产品发布, 大模型适配, 股东变动)
- Switch to "资讯" tab by clicking the link for news-only view

## Analysis Framework

Present analysis in this structured format:

### 📊 核心数据 (Core Data)
Table with: price, change%, open/high/low, volume, turnover, market cap, PE ratio

### ⚠️ 风险警示信号 (Risk Signals)
After presenting core data, highlight risks upfront:
- **异常波动公告**: 近期是否触发连续三日涨跌幅偏离值达20%的公告
- **极高换手率**: >5% 活跃，>7% 可能有资金出逃
- **估值警示**: PE > 100 需提醒高估风险
- **大盘风险**: 上证/深证大跌时系统性风险主导

### 📰 近期重要动态 (Recent Developments)
Bullet list of news, announcements, catalysts

### 🔍 技术面分析 (Technical Analysis)

| 维度 | 判断 |
|------|------|
| 短线趋势 (Short-term) | ⚠️/✅/❌ |
| 成交量 (Volume) | 放量/缩量 + direction |
| 换手率 (Turnover) | H/M/L |
| 振幅 (Amplitude) | H/M/L |
| 市盈率 (PE) | relative to sector |

### 🎯 每日交易建议 (Daily Trading Advice)

**⚠️ 重要声明:** Include a disclaimer that this is not investment advice.

**卖出建议 (Sell):**
- 压力位 (Resistance) — 1-2 key levels
- 止损位 (Stop-loss) — breach = exit

**买入建议 (Buy):**
- 支撑位 (Support) — 1-2 key levels for entry
- 建仓策略 — staged entry plan

**仓位控制 (Position Sizing):**
- Recommended % of total capital given current market conditions

**次日策略表 (Next-day Strategy Table):**

| 场景 (Scenario) | 操作建议 (Action) |
|-----------------|-------------------|
| 高开至XX+ | 逢高减仓，不追 |
| 平开震荡XX-XX | 观察，不操作 |
| 低开跌破XX | 止损或观望 |
| 深跌至XX-XX | 轻仓试探性建仓 |

### 趋势跟踪指标 (Trend Tracking)
Key signals to watch: index trend, sector trend, volume contraction, policy catalysts

## Quantitative Analysis via execute_code

After collecting real-time data from 东方财富 API and historical K-line data from the 腾讯API, use `execute_code` with Python for deep statistical analysis. The JSON APIs return raw data directly — no browser needed.

### Workflow

1. **Get historical data** from 腾讯API:
   ```bash
   curl -s "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh{CODE},day,,,120,qfq" -H "User-Agent: Mozilla/5.0"
   ```
2. **Get real-time quote** from 东方财富 Push API:
   ```bash
   curl -s "https://push2.eastmoney.com/api/qt/stock/get?secid=1.{CODE}&fields=f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162,f167,f168,f169,f170,f171" -H "User-Agent: Mozilla/5.0"
   ```
3. **Parse both JSON outputs** in a Python `execute_code` script
4. **Extract key patterns** and incorporate into the daily trading advice

### Key Analyses to Run

| Analysis | What It Reveals |
|----------|----------------|
| **收益率统计** (Return Stats) | Avg daily return, max up/down, win rate |
| **星期几效应** (Weekday Effect) | Strongest/weakest weekday |
| **连续涨跌分析** (Streak Analysis) | Max up/down streak, current streak |
| **暴跌反弹统计** (Drop Recovery) | After -3%+ drops, next-day bounce probability (~60% typical) |
| **暴涨持续统计** (Rise Persistence) | After +3%+ rises, next-day continuation rate (~74% typical) |
| **成交量分析** (Volume Analysis) | Avg volume, spike days, volume pattern at turning points |
| **均线计算** (MA Calculation) | MA5/10/20/30 — compare current price against them |
| **波浪周期识别** (Wave Cycle) | Identify bull/bear/sideways phases from local tops/bottoms |
| **支撑/压力位** (S/R Levels) | Quartile distribution of historical lows/highs |

### Reusable Script

See `references/quantitative-analysis.py` — a complete Python script that does all of the above. Paste 腾讯API JSON data into the `RAW_JSON` variable and run in `execute_code`.

### Discovered Patterns (from this session's 摩尔线程 analysis)

#### 📅 Weekday Effect
| Day | Avg Return | Win Rate | Action |
|-----|-----------|----------|--------|
| 周一 | +0.06% | 50.0% | Neutral |
| **周二** | **+0.55%** 🟢 | **66.7%** | **Best day — buy bias** |
| 周三 | +0.21% | 57.7% | Mildly bullish |
| 周四 | +0.14% | 44.0% | Start reducing |
| **周五** | **-0.78%** 🔴 | **30.4%** | **Worst day — sell/reduce** |

**Rule**: Validate per stock, but this pattern was strong on a high-vol 科创板 stock.

#### 📊 暴跌反弹模式
- After -3%+ drop: **60% probability** of next-day bounce, avg +0.75%
- After -5%+ drops: bounce probability even higher
- **Rule**: Do NOT panic-sell on big red days; wait for next-day bounce

#### 📈 暴涨持续模式
- After +3%+ rise: **74% probability** of next-day continuation, avg +1.05%
- **Rule**: Do NOT sell on big green days; momentum carries into next session

#### 🔄 波浪周期策略
| Phase | Sign | Strategy |
|-------|------|----------|
| **探底期** | Lower highs/lows, MA5<MA10<MA20 | Avoid, wait for base |
| **主升浪** | Breaking MA20, expanding volume | Trend-follow, buy pullbacks |
| **高位震荡** | Range-bound, declining volume | Range-trade, buy support/sell resistance |

### Example execute_code Script Template

```python
from hermes_tools import terminal
import json

# Step 1: Fetch real-time quote
r1 = terminal("curl -s 'https://push2.eastmoney.com/api/qt/stock/get?secid=1.688795&fields=f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162,f167,f168,f169,f170,f171' -H 'User-Agent: Mozilla/5.0'", timeout=10)
quote = json.loads(r1['output'])['data']
price = quote['f43'] / 100
change = quote['f55'] * 100
print(f"Price: {price}, Change: {change:.2f}%")

# Step 2: Fetch historical K-line
r2 = terminal("curl -s 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh688795,day,,,120,qfq' -H 'User-Agent: Mozilla/5.0'", timeout=10)
kdata = json.loads(r2['output'])
days = kdata['data']['sh688795']['day']

# Step 3: Parse and analyze
closes = [float(d[2]) for d in days]
for n in [5, 10, 20, 30]:
    ma = sum(closes[-n:])/n
    print(f"MA{n}: {ma:.2f} ({'above' if closes[-1]>ma else 'below'})")
```

## Pitfalls

1. **大盘风险**: When the broader market (上证/深证) is down heavily (-2%+), individual stock analysis is secondary — systematic risk dominates. Advise defensive positioning.
2. **次新股/科创板**: High PE (2000+) stocks on 科创板 with small float relative to total cap are extremely volatile and speculative. Emphasize strict stop-losses.
3. **盘后数据时效**: After 15:00 CST the market is closed. Next-day guidance is based on closing data and cannot account for after-hours events.
4. **No live trading**: The agent cannot execute trades or connect to brokerages. All advice is informational only.
5. **High turnover warning**: Turnover rate >7% with falling price = distribution / institutional selling.
6. **流通市值 vs 总市值**: Big gap means limited float — easier to manipulate, higher volatility.

## Stock Code Quick Reference (Common Prefixes)

| Board | Prefix | Example |
|-------|--------|---------|
| 上证主板 | 600/601/603 | 600519 (贵州茅台) |
| 深证主板 | 000/002 | 000858 (五粮液) |
| 创业板 | 300 | 300750 (宁德时代) |
| 科创板 | 688 | 688981 (中芯国际) |
| 北交所 | 8xx | - |

## Daily Monitoring via Cron Job

For recurring daily stock monitoring (用户要求"每天给交易建议"), use a cron job that runs after market close.

### Setup Pattern

1. **Create a context file** (`context.md`) in the project directory with:
   - Task configuration (stock code, data sources)
   - Latest snapshot
   - Technical baseline (MAs, S/R levels)
   - Weekday effect stats
   - Historical record section

2. **Create a cron job** with:
   - Schedule: `30 7 * * 1-5` (weekdays at 7:30 CST, before market open)
   - Workdir: the project directory
   - Enabled toolsets: `terminal`, `file`, `web`
   - Prompt: instructs the agent to fetch quote → fetch K-line → compute indicators → output structured report → update context.md

3. **Report format** (structured, per user preference):
   ```
   ## 📊 核心数据 (table)
   ## ⚠️ 风险警示
   ## 📰 近期动态
   ## 🔍 技术面分析
   ## 🎯 今日交易策略 (scenario table)
   ```

### Cron Job Prompt Template

```markdown
执行{STOCK_NAME}({CODE})的每日收盘分析任务。

### 1. 获取实时行情
curl -s "https://push2.eastmoney.com/api/qt/stock/get?secid=1.{CODE}&fields=f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162,f167,f168,f169,f170,f171" -H "User-Agent: Mozilla/5.0"

### 2. 获取历史K线
curl -s "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh{CODE},day,,,120,qfq" -H "User-Agent: Mozilla/5.0"

### 3. 计算技术指标
- MA5/10/20/30, 支撑/压力位
- 星期几效应判断

### 4. 输出结构化报告并更新 context.md
```

### ⚠️ Pitfall: TUI Delivery
When running in the Hermes TUI (terminal UI), cron jobs default to `deliver='local'` — output is saved but NOT pushed to the session. If the user wants push notifications, set `deliver='telegram'` or `deliver='all'` targeting a gateway-connected platform.

## References

- `references/eastmoney-quote-fields.md` — Field mapping for 东方财富 quote pages
- `references/eastmoney-push-api-fields.md` — Complete field code reference for the Push API
- `references/technical-indicators.md` — Technical analysis indicator reference
- `references/quantitative-analysis.py` — Reusable Python script for full statistical analysis (weekday effect, drop recovery, rise persistence, MAs, wave cycles, S/R levels). Paste data from 腾讯API output and run in execute_code.
