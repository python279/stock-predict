# 全球新闻自动分析系统

## 📋 项目简介

完全自动化的国际新闻抓取、分析和通知系统。系统每天自动抓取世界各地的权威新闻源，使用大模型进行深度分析，识别**黑天鹅事件**、预测世界局势并分析 **A股/美股/港股/日股** 投资机会，通过邮件自动发送精美 HTML 格式分析报告。

## ✨ 主要特性

- 🌍 **多源新闻抓取**: News API、RSS、GDELT DOC API 与 Google News 搜索 RSS；聚合源通过白名单过滤
- ⚡ **黑天鹅预警**: 对比可配置的历史新闻窗口，识别军事/金融/政治/能源/科技突发信号
- 🤖 **智能AI分析**: system/user prompt 分离，强制 Markdown 输出格式
- 📈 **四大市场覆盖**: A股、美股、港股、日本股市专项分析
- 📉 **行业数据增强**: 注入美股指数/ETF与A股宽基/行业行情，覆盖科技（芯片、AI、存储）、金融和大消费
- 📰 **行业证据包**: 新闻自动打行业/政策标签，结合权威政策新闻和透明的轻量舆情摘要
- 🏛️ **美国政治传导**: 跟踪选举、白宫/国会政策、制裁与外交信号，区分事实、传导假设和待验证市场信号
- 🛢️ **商品趋势数据**: 可选接入 Yahoo Finance 日线，分析油、金、铜、农产品等的趋势与跨资产传导
- 💰 **投资策略**: 针对黑天鹅情景给出收益最大化方案（含 T+0~T+3 快速响应）
- 📧 **自动邮件**: 每天定时发送，Markdown 完整渲染（表格/粗体/引用块）
- ⏰ **自动调度**: 支持定时自动执行，完全无人值守
- 🛡️ **高可靠性**: 完善的错误处理和重试机制
- ⚙️ **高度可配置**: 所有参数均可通过配置文件调整

## 🏗️ 项目结构

```
info-os/
├── config.example.yaml      # 可提交的配置模板
├── config.yaml              # 本地主配置文件（由模板复制，勿提交）
├── requirements.txt         # Python依赖
├── README.md                # 本文件
├── run.sh                   # 快速启动脚本
├── src/                     # 源代码目录
│   ├── config_loader.py     # 配置加载模块
│   ├── news_fetcher.py      # 新闻抓取模块（News API + RSS + GDELT + Google News）
│   ├── commodity_fetcher.py # 大宗商品行情与趋势指标
│   ├── market_data_fetcher.py # 美股/A股行情与趋势指标
│   ├── sentiment_fetcher.py # 行业舆情摘要（新闻为主、股吧可选）
│   ├── llm_analyzer.py      # 大模型分析模块（system/user prompt 分离）
│   ├── email_sender.py      # 邮件发送模块（markdown→HTML）
│   ├── data_storage.py      # 数据存储模块（含历史新闻加载）
│   ├── logger_config.py     # 日志配置模块
│   ├── main.py              # 主程序
│   ├── scheduler.py         # 定时调度器
│   ├── test_config.py       # 配置测试工具
│   ├── test_rss_sources.py  # RSS源测试工具
│   └── test_email.py        # 邮件测试工具
│   └── test_market_data.py  # 市场、限流、提示词与格式测试
├── data/                    # 数据目录
│   ├── news_cache/          # 新闻缓存（供历史分析使用）
│   ├── market_cache/        # 市场和舆情快照
│   ├── reports/             # 完整分析报告（JSON + Markdown）
│   └── mobile_reports/      # 移动版 Markdown 报告
└── logs/                    # 日志目录
    └── news_analyzer.log
```

## 🚀 快速开始（5分钟）

### Python 解释器说明

**本项目默认使用虚拟环境的 Python 解释器：**
- 路径: `.venv/bin/python3`
- 所有命令示例均使用此解释器
- 建议使用虚拟环境以避免依赖冲突

### 步骤 0: 创建本地配置

```bash
cp config.example.yaml config.yaml
```

随后编辑 `config.yaml` 填入邮箱及模型密钥。该文件包含敏感信息，不应提交。

### 步骤 1: 安装依赖

**创建并激活虚拟环境：**
```bash
cd /path/to/info-os

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

**或直接使用完整路径：**
```bash
.venv/bin/pip install -r requirements.txt
```

### 步骤 2: 配置邮箱（⚠️ 重要）

系统支持多种邮箱服务，推荐使用QQ邮箱（国内访问稳定）。

#### 选项A: 使用QQ邮箱（推荐）

1. **开启SMTP服务**
   - 登录 QQ 邮箱网页版
   - 设置 → 账户 → POP3/IMAP/SMTP服务
   - 开启 SMTP 服务
   - 生成授权码（记住这个授权码，不是QQ密码）

2. **更新配置文件** `config.yaml`：
```yaml
email:
  smtp_server: "smtp.qq.com"
  smtp_port: 587
  sender_email: "your-qq-number@qq.com"
  sender_password: "授权码"  # 生成的授权码
  recipient_emails:          # 支持多个收件人
    - "your-qq-number@qq.com"
    - "another@163.com"
```

#### 选项B: 使用Gmail

1. **启用两步验证**：访问 https://myaccount.google.com/security

2. **生成应用专用密码**：访问 https://myaccount.google.com/apppasswords

3. **更新配置文件**：
```yaml
email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "your-email@gmail.com"
  sender_password: "xxxx xxxx xxxx xxxx"  # 应用专用密码
  recipient_emails:
    - "your-email@gmail.com"
```

> **注意**: Gmail需要使用应用专用密码，不是登录密码。如果连接超时，建议切换到QQ邮箱。

#### 其他邮箱

```yaml
# 163邮箱
email:
  smtp_server: "smtp.163.com"
  smtp_port: 465
  sender_email: "your-email@163.com"
  sender_password: "授权码"
```

### 步骤 3: 配置大模型API

系统支持多种大模型，选择其中一种配置即可。

#### 本地 Ollama（默认，免费）
```yaml
llm:
  provider: "openai-compatible"
  api_key: ""
  model: "qwen3.5:35b"  # 或其他本地模型
  base_url: "http://localhost:11434/v1/"
  max_tokens: 65536
  temperature: 0.8
  timeout_seconds: 1800
  max_retries: 1
```

#### OpenAI（GPT-4）
```yaml
llm:
  provider: "openai"
  api_key: "sk-..."
  model: "gpt-4o"
  base_url: ""
  max_tokens: 8000
  temperature: 0.7
  timeout_seconds: 1800
  max_retries: 1
```

#### OpenAI兼容接口（讯飞星火、通义千问等）
```yaml
llm:
  provider: "openai-compatible"
  api_key: "your-api-key"
  model: "x1"
  base_url: "https://spark-api-open.xf-yun.com/v2/"
  max_tokens: 32768
  temperature: 0.7
  timeout_seconds: 1800
  max_retries: 1
```

#### Anthropic Claude
```yaml
llm:
  provider: "anthropic"
  api_key: "sk-ant-..."
  model: "claude-3-5-sonnet-20241022"
  max_tokens: 8000
  temperature: 0.7
  timeout_seconds: 1800
  max_retries: 1
```

### 步骤 4: 配置News API（可选）

访问 https://newsapi.org 注册获取免费 API Key（100 次/天）：

```yaml
news_api:
  enabled: true
  api_key: "your-newsapi-key"
```

> `countries × categories` 决定每轮 News API 请求数；模板默认约 30 次。HTTP 429 不会重试，程序会停止该源本轮请求并继续其他来源。

### 步骤 4.5: 配置免费聚合新闻源（可选）

`config.example.yaml` 默认启用 GDELT 与 Google News 搜索 RSS，二者都不需要 API Key。请维护 `trusted_domains` / `trusted_sources` 白名单，只将可信媒体的标题级聚合结果送入报告。模板还通过 Google News 的 IMF 官方域名检索补充 IMF 新闻。

> 不配置 News API 也可运行：RSS、GDELT、Google News 与 IMF 归档会继续工作。GDELT 遇 HTTP 429 时同样停止本轮后续查询；其他网络错误按 `fetcher.retry_*` 重试。

### 步骤 5: 测试配置

```bash
.venv/bin/python3 src/test_config.py

# 市场数据、限流、提示词和邮件格式的离线测试
.venv/bin/python3 -m unittest discover -s src -p 'test_market_data.py'
```

### 步骤 6: 立即运行

**立即执行一次（推荐先试试）：**
```bash
.venv/bin/python3 src/main.py
```

系统会：
1. 抓取新闻（News API、RSS、GDELT、Google News、IMF 官网归档；耗时取决于网络与启用源）
2. 可选抓取大宗商品、美股/A股市场行情与行业舆情摘要
3. 保存新闻和市场缓存，加载历史新闻窗口（示例默认 14 天、每日最多 25 篇）
4. AI 生成市场与行业报告，再保存并发送 HTML 邮件

**启动定时调度器：**
```bash
.venv/bin/python3 src/scheduler.py
```

**使用启动脚本（最简单）：**
```bash
./run.sh
```

---

## 📰 新闻源配置

以下是推荐的权威 RSS 扩展清单。`config.example.yaml` 默认启用其中 20 个 RSS 源，并额外启用 Federal Reserve Press；可根据网络条件和需要补齐或替换。

### 🔴 电报社 / 突发预警（4个）
突发事件比普通媒体**早15-30分钟**报道，黑天鹅预警核心来源。

| # | 名称 | 特点 |
|---|------|------|
| 1 | **Reuters World** | 全球最大通讯社 |
| 2 | **Sky News World** | 英国24小时滚动 |
| 3 | **NBC News** | 美国主流突发 |
| 4 | **ABC News International** | 美国主流突发 |

### 🌐 全球综合（3个）

| # | 名称 | 特点 |
|---|------|------|
| 5 | **BBC World News** | 英国公共广播 |
| 6 | **BBC Business** | BBC商业频道 |
| 7 | **Al Jazeera** | 中东视角 |

### 🗺️ 区域媒体（16个）

**欧洲（4个）**: Financial Times、Deutsche Welle、France 24、The Guardian World

**亚洲（6个）**: Channel NewsAsia、Nikkei Asia、The Straits Times、Bangkok Post、South China Morning Post、Asia Times

**俄罗斯（2个）**: TASS、Moscow Times

**美洲（4个）**: NPR News、CNN Top Stories、CNN World（+NBC/ABC已在突发预警分类）

### 🔥 黑天鹅专项（4个）
专注于地缘政治高风险区域和军事动态。

| # | 名称 | 专注领域 |
|---|------|---------|
| 24 | **Middle East Eye** | 中东事件，以色列/伊朗冲突最快报道 |
| 25 | **Jerusalem Post** | 以色列视角，中东军事预警 |
| 26 | **IFP News (Iran)** | 伊朗视角，了解对方行动信号 |
| 27 | **Defence Blog** | 军事装备、军事行动实时 |

### 💹 金融/政治深度（5个）

| # | 名称 | 专注领域 |
|---|------|---------|
| 28 | **Defense News** | 美国防务政策与军事采购 |
| 29 | **CNBC World News** | 突发对市场影响第一时间 |
| 30 | **Washington Post World** | 华盛顿决策内幕 |
| 31 | **NYT World** | 深度背景报道 |
| 32 | **UN News / IMF 官方新闻** | 国际组织官方动态 |

### 🏛️ 官方政策源

- **Federal Reserve Press**：示例配置默认启用，用于补充美国货币政策和金融监管的一手信息。
- **IMF 官方新闻**：Google News RSS 仅接收 `International Monetary Fund | IMF` 的官方结果；同时通过 IMF “What's New Archive”的官方动态 JSON 接口抓取更新。接口返回 HTTP 403、429 或暂时不可用时会跳过归档源并继续其他新闻源，不尝试绕过访问限制。
- 中国政策信息应优先从国务院、人民银行、证监会、工信部等官方网站获取；请仅在确认 RSS/API 稳定且符合使用条款后加入 `rss_feeds.sources`。

### 测试RSS源

```bash
.venv/bin/python3 src/test_rss_sources.py
```

### 免费新闻源与限流降级

除 RSS 外，系统支持无需密钥的 GDELT DOC API、Google News 搜索 RSS 与 IMF “What's New Archive”归档。GDELT 与 Google News 均通过配置白名单过滤媒体来源，避免把聚合站或未知站点直接送入报告。NewsAPI 返回 HTTP 429 时，程序会立即停止该轮的后续 NewsAPI 请求，继续使用 RSS、GDELT、Google News 和 IMF 归档，不会进行无效重试。

默认查询还包含美国选举、白宫、国会、外交与中东政策热点。政治新闻只用于构建“政策预期 → 经济/资产”的可验证传导假设；不将时间相关性直接表述为因果关系。

---

## 📊 分析报告结构

报告固定为 7 个主章节，重要结论、风险与操作前置；宏观、商品和跨资产传导合并，删除容易与黑天鹅策略重复的独立高收益章节。关闭 `include_a_share_analysis` 时，第 4 章仅保留中国市场观察。

### 第1章：今日决策摘要
用表格前置最重要事件、市场影响、当前操作倾向和需立即关注的触发条件。

### 第2章：关键风险与黑天鹅监测
仅保留 2–3 个最高优先级情景，合并地缘政治、金融、供应链和科技风险，并给出触发条件与 T+0 至 T+3 防守响应。

### 第3章：宏观与跨资产传导
把增长、央行、贸易和大宗商品合并为一张传导矩阵，只讨论有数据或新闻支持的变量。

### 第4章：A股行业趋势与科技急跌预警
覆盖科技（芯片、AI、存储）、金融和大消费。科技急跌部分是基于相对宽基表现、回撤、均线、量能与新闻压力的脆弱性预警，不是黑天鹅确定性预测。

### 第5章：美股行业与风险偏好
结合 SPY/QQQ、行业 ETF、VIX、政策与新闻，分析科技、金融和消费。

### 第6章：港股与日股联动观察
聚焦与 A股/美股的联动、汇率、资金流和供应链差异；无新增高置信度信号时会明确说明。

### 第7章：未来路径与执行清单
给出未来 5 个交易日和 1–4 周的基准/上行/风险情景、触发条件、观察项与风险警戒线。

**报告保存位置：**
- JSON格式: `data/reports/report_*.json`
- Markdown 格式: `data/reports/report_*.md`
- 移动版 Markdown: `data/mobile_reports/mobile_report_*.md`（移除新闻参考资料；表格转为逐项列表，适合手机窄屏阅读）
- 邮件: 完整 HTML 渲染（表格/粗体/引用块等）

> 表格单元格仅使用简短文本，多个条件以 `；` 分隔。邮件渲染会自动将模型偶发输出的 LaTeX 箭头和单元格内行内列表规范化，避免显示为代码或拥挤文本。

---

## ⚙️ 高级配置

### 分析配置（`config.yaml`）

```yaml
analysis:
  focus_areas:
    - "全球经济趋势"
    - "地缘政治风险"
    - "黑天鹅事件预警与投资策略"
    - "中国A股市场分析"
    - "美股市场分析（标普500/纳斯达克/道琼斯）"
    - "港股市场分析（恒生指数/恒生科技）"
    - "日本股市分析（日经225/日元汇率）"
    - "大宗商品价格（原油/黄金/铜）"
    - "AI行业发展"
    - "半导体产业链"
  output_language: "zh-CN"
  include_predictions: true
  prediction_timeframe: "未来1-3个月"
  include_a_share_analysis: true
  short_term_timeframes: ["未来5个交易日", "未来1-4周"]
  a_share_focus:
    - "沪深300指数趋势"
    - "创业板/科创板机会"
    - "行业板块轮动"
    - "政策导向行业"
    - "出口导向企业影响"
    - "资金流向分析"
    - "芯片/AI/存储产业链"
    - "银行/券商/保险"
    - "必需消费与可选消费"
```

### 市场行情和舆情配置

`markets.assets` 按 `provider` 配置数据源：美股与部分 A 股 ETF 可使用 Yahoo Finance；A 股宽基优先使用腾讯实时快照和日线，腾讯不可用时才尝试东方财富备用快照，需提供 `tencent_symbol`、`secid`。程序生成 1/5/20 日涨跌、MA5/20/60、20 日回撤、滚动波动率、量能比、趋势与行业风险等级；行情失败时报告会明确降级为新闻情景分析。

`sentiment` 默认仅汇总已抓取的权威新闻标题和市场相对强弱。`guba_enabled` 默认为 `false`，`guba_codes` 可按 `tech`、`finance`、`consumer` 配置代码列表；启用前请确认数据源条款与使用场景。

`storage.historical_news_days` 和 `storage.historical_news_max_per_day` 分别控制历史新闻窗口及每日输入上限；示例默认 14 天和 25 篇。更长窗口会增加模型上下文与调用成本。

`commodities` 是可选模块，使用 Yahoo Finance 日线计算商品趋势。启用后可配置 `historical_days`、超时/重试、`assets` 与按市场映射的 `investment_targets`；未配置时商品章节会保留但明确标记为数据缺失。

### 调度配置

```yaml
scheduler:
  enabled: true
  run_time: "08:00"       # 每天固定运行时间（24小时制）
  timezone: "Asia/Shanghai"
```

当前 `scheduler.py` 同时会在每天 `run_time` 和每小时整点触发任务。若只需每天一次，请使用 Cron 调度 `main.py`，或修改调度代码后再启动内置调度器。`enabled: false` 时会立即执行一次后退出。

### 内容过滤

```yaml
content_filter:
  min_content_length: 100
  exclude_keywords:
    - "horoscope"
    - "celebrity gossip"
    - "entertainment"
    - "sports"
  include_keywords:
    # 经济类
    - "economy"
    - "market"
    - "trade"
    - "inflation"
    # 黑天鹅类（新增）
    - "strike"
    - "attack"
    - "military"
    - "missile"
    - "conflict"
    - "war"
    - "crisis"
    - "coup"
    - "nuclear"
    - "escalation"
  sector_keywords:
    tech: ["AI", "semiconductor", "chip", "memory", "storage", "芯片", "半导体"]
    finance: ["bank", "credit", "interest rate", "银行", "券商", "保险"]
    consumer: ["consumer", "retail", "消费", "零售", "社零"]
    policy: ["policy", "regulation", "政策", "监管", "证监会", "人民银行"]
```

### 添加新的新闻源

编辑 `config.yaml`，在 `rss_feeds.sources` 中添加：

```yaml
- name: "新的新闻源"
  url: "https://example.com/rss"
  region: "middle_east"  # global, americas, europe, asia, russia, middle_east, international
  priority: 5            # 1-5，数字越大优先级越高
```

---

## 🛠️ 运维管理

### 查看日志

```bash
# 实时查看日志
tail -f logs/news_analyzer.log

# 只看错误
grep ERROR logs/news_analyzer.log
```

### 查看报告

```bash
# 查看最新 Markdown 报告
ls -lt data/reports/*.md | head -3

# 预览最新报告
cat $(ls -t data/reports/*.md | head -1) | head -100

# 查看最新移动版报告
cat $(ls -t data/mobile_reports/*.md | head -1) | head -100
```

### 测试邮件发送

```bash
.venv/bin/python3 src/test_email.py
```

### 清理缓存

系统会自动清理超过配置天数的缓存，也可手动清理：

```bash
rm -rf data/news_cache/*
rm -rf data/market_cache/*
```

> ⚠️ 注意：手动清除后，下次运行将没有历史新闻参考数据，黑天鹅对比分析效果会下降。

### 后台运行调度器

```bash
nohup .venv/bin/python3 src/scheduler.py > logs/scheduler.log 2>&1 &

# 查看进程
ps aux | grep scheduler

# 停止
kill <PID>
```

### 使用 Cron 定时运行

```bash
crontab -e
```

添加（每天8:00运行）：
```
0 8 * * * cd /path/to/info-os && .venv/bin/python3 src/main.py >> logs/cron.log 2>&1
```

---

## 🔍 常见问题

### Q1: 邮件中的表格/粗体没有正常显示？

系统使用 `markdown` 库将 LLM 输出转换为 HTML。如果某些格式未渲染：
- 检查 LLM 是否正确输出了 Markdown（查看 `data/reports/report_*.md`）
- 如果 LLM 将整个回复包在 ` ```markdown ``` ` 代码块里，系统会自动剥除
- 表格内不要使用 `- `、`1.` 或换行列表；多个条件用 `；`。渲染器会对历史报告中的行内列表做兼容清理
- 确认 `requirements.txt` 中的 `markdown>=3.5` 已安装

### Q2: 邮件发送失败？

**Gmail超时**: 网络无法访问 Gmail SMTP，建议切换到 QQ 邮箱

**认证失败**: 确保使用授权码/应用专用密码，不是登录密码

```bash
.venv/bin/python3 src/test_email.py
```

### Q3: 黑天鹅章节内容空洞？

黑天鹅分析依赖历史新闻对比。如果是首次运行或缓存被清除，历史数据为空，分析质量较低。建议运行几天后效果明显提升。

### Q4: 大模型API调用失败？

- 检查 `config.yaml` 中 `api_key`、`base_url`、`model` 是否正确
- 本地 Ollama：确认服务已启动（`ollama serve`）且模型已下载（`ollama pull qwen3.5:35b`）
- 检查 `max_tokens` 是否超出模型限制
- 本地长上下文报告可调大 `timeout_seconds`（示例为 1800 秒）；`max_retries` 控制 SDK 的自动重试次数

### Q5: 没有抓取到新闻？

```bash
# 测试 RSS 源可用性
.venv/bin/python3 src/test_rss_sources.py

# 禁用所有聚合/API 新闻源，仅使用 RSS
# config.yaml: news_api.enabled: false
#              gdelt.enabled: false
#              google_news.enabled: false
```

### Q6: 报告内容质量不满意？

- 更换更强大的模型（如 GPT-4o 或 Claude-3.5-Sonnet）
- 调整 `config.yaml` 的 `analysis.focus_areas`
- 增加 `max_tokens` 以获得更长更详细的分析

---

## 💰 成本估算

### 免费资源
- ✅ RSS Feed、GDELT、Google News RSS: 无需 API Key
- ✅ News API: 免费版 100 次/天；默认配置每轮约 30 次请求
- ✅ 邮箱服务: 完全免费
- ✅ 本地 Ollama: 完全免费（需要本地GPU/CPU算力）

### 付费资源（按每次分析估算）
| 模型 | 每次费用 | 月费用（每天1次） |
|------|---------|----------------|
| 本地 Ollama | 免费 | 免费 |
| 讯飞星火 | ~¥0.03-0.06 | ~¥1-2 |
| GPT-4o | ~¥0.3-0.6 | ~¥10-18 |
| Claude 3.5 Sonnet | ~¥0.2-0.4 | ~¥6-12 |

> 报告通常含 12 个主章节与可选商品专节。实际 Token 消耗受历史窗口、启用数据和模型输出长度影响，应以模型服务的用量统计为准。

---

## ⚠️ 注意事项

### 1. 隐私安全
- ⚠️ **不要将 `config.yaml` 提交到公开仓库**（包含 API Key 和邮箱密码）
- 已在 `.gitignore` 中配置忽略

### 2. API 限制
- News API 免费版：100请求/天
- 邮箱：避免频繁发送（建议每天1次）

### 3. 历史新闻缓存
- 历史新闻缓存存于 `data/news_cache/`，市场和舆情快照存于 `data/market_cache/`
- 默认保留360天，可通过 `storage.max_cache_days` 调整
- 清除缓存会降低黑天鹅分析精度

### 4. 法律合规
- 仅用于个人学习和研究
- 遵守各新闻源的使用条款
- 不用于商业目的或二次分发

---

## 🤝 扩展开发

### 添加新的通知方式

扩展 `email_sender.py`：
- Telegram Bot
- 企业微信 Webhook
- 钉钉通知
- Slack

### 自定义分析维度

修改 `config.yaml` 的 `analysis.focus_areas`，或修改 `src/llm_analyzer.py` 的 `_build_system_prompt()` 调整报告章节结构。

### 集成实时社交媒体（进阶）

目前已有 RSS、GDELT 和 Google News 的标题级聚合。如需进一步扩展实时信号，可集成：
- **Twitter/X API**（付费，$100/月 Basic）：监控 `@Reuters`、`@OSINTdefender` 等
- **Telegram Bot API**（免费）：订阅 `@warmonitors`、`@intelrepublic` 等频道

---

## 📝 维护建议

### 定期检查（每月一次）

1. **测试RSS源状态**
```bash
.venv/bin/python3 src/test_rss_sources.py
```

2. **检查API配额**：News API 使用量、大模型 API 费用

3. **审查报告质量**：重点检查黑天鹅章节的信号识别是否准确

4. **查看日志异常**
```bash
grep ERROR logs/news_analyzer.log | tail -20
```

### 备份重要数据

```bash
# 备份配置
cp config.yaml config.yaml.backup

# 备份历史报告
tar -czf reports_backup_$(date +%Y%m%d).tar.gz data/reports/
```

---

## 📄 许可证

MIT License - 可自由使用、修改和分发

---

## 🎉 开始使用

1. ✅ 创建虚拟环境: `python3 -m venv .venv`
2. ✅ 安装依赖: `.venv/bin/pip install -r requirements.txt`
3. ✅ 配置邮箱: 编辑 `config.yaml`（推荐QQ邮箱）
4. ✅ 配置大模型: 填入API Key 或配置本地 Ollama
5. ✅ 测试配置: `.venv/bin/python3 src/test_config.py`
6. ✅ 立即运行: `.venv/bin/python3 src/main.py`
7. ✅ 启动调度: `.venv/bin/python3 src/scheduler.py`

**⚡ 让AI帮你每天洞察世界，识别黑天鹅，把握四大市场投资机会！**
