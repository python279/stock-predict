# 全球新闻自动分析系统

## 📋 项目简介

完全自动化的国际新闻抓取、分析和通知系统。系统每天自动抓取世界各地的权威新闻源，使用大模型进行深度分析，识别**黑天鹅事件**、预测世界局势并分析 **A股/美股/港股/日股** 投资机会，通过邮件自动发送精美 HTML 格式分析报告。

## ✨ 主要特性

- 🌍 **多源新闻抓取**: News API + **32个**权威 RSS 新闻源（含黑天鹅专项渠道）
- ⚡ **黑天鹅预警**: 对比可配置的历史新闻窗口，识别军事/金融/政治/能源/科技突发信号
- 🤖 **智能AI分析**: system/user prompt 分离，强制 Markdown 输出格式
- 📈 **四大市场覆盖**: A股、美股、港股、日本股市专项分析
- 📉 **行业数据增强**: 注入美股指数/ETF与A股宽基/行业行情，覆盖科技（芯片、AI、存储）、金融和大消费
- 📰 **行业证据包**: 新闻自动打行业/政策标签，结合权威政策新闻和透明的轻量舆情摘要
- 💰 **投资策略**: 针对黑天鹅情景给出收益最大化方案（含 T+0~T+3 快速响应）
- 📧 **自动邮件**: 每天定时发送，Markdown 完整渲染（表格/粗体/引用块）
- ⏰ **自动调度**: 支持定时自动执行，完全无人值守
- 🛡️ **高可靠性**: 完善的错误处理和重试机制
- ⚙️ **高度可配置**: 所有参数均可通过配置文件调整

## 🏗️ 项目结构

```
info-os/
├── config.yaml              # 主配置文件
├── requirements.txt         # Python依赖
├── README.md                # 本文件
├── run.sh                   # 快速启动脚本
├── src/                     # 源代码目录
│   ├── config_loader.py     # 配置加载模块
│   ├── news_fetcher.py      # 新闻抓取模块（News API + RSS）
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
├── data/                    # 数据目录
│   ├── news_cache/          # 新闻缓存（供历史分析使用）
│   ├── market_cache/        # 市场和舆情快照
│   └── reports/             # 分析报告（JSON + TXT）
└── logs/                    # 日志目录
    └── news_analyzer.log
```

## 🚀 快速开始（5分钟）

### Python 解释器说明

**本项目默认使用虚拟环境的 Python 解释器：**
- 路径: `.venv/bin/python3`
- 所有命令示例均使用此解释器
- 建议使用虚拟环境以避免依赖冲突

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
```

#### Anthropic Claude
```yaml
llm:
  provider: "anthropic"
  api_key: "sk-ant-..."
  model: "claude-3-5-sonnet-20241022"
  max_tokens: 8000
  temperature: 0.7
```

### 步骤 4: 配置News API（可选）

访问 https://newsapi.org 注册获取免费API Key（100次/天）：

```yaml
news_api:
  enabled: true
  api_key: "your-newsapi-key"
```

> **不配置也可以**：系统会自动使用32个RSS新闻源，完全够用。

### 步骤 5: 测试配置

```bash
.venv/bin/python3 src/test_config.py
```

### 步骤 6: 立即运行

**立即执行一次（推荐先试试）：**
```bash
.venv/bin/python3 src/main.py
```

系统会：
1. 抓取全球新闻（约60秒，32个RSS源）
2. 抓取配置的美股/A股市场行情与行业舆情摘要
3. 加载可配置的历史新闻窗口（默认14天，用于黑天鹅对比分析）
4. AI分析生成市场与行业报告（依模型而定）
4. 发送HTML邮件报告

**启动定时调度器（每天自动运行）：**
```bash
.venv/bin/python3 src/scheduler.py
```

**使用启动脚本（最简单）：**
```bash
./run.sh
```

---

## 📰 新闻源配置（32个）

系统配置了32个权威新闻源，按功能分为五大类。

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
| 32 | **UN News / IMF News** | 国际组织官方动态 |

### 测试RSS源

```bash
.venv/bin/python3 src/test_rss_sources.py
```

### 免费新闻源与限流降级

除 RSS 外，系统支持无需密钥的 GDELT DOC API 与 Google News 搜索 RSS。两者均通过配置白名单过滤媒体来源，避免把聚合站或未知站点直接送入报告。NewsAPI 返回 HTTP 429 时，程序会立即停止该轮的后续 NewsAPI 请求，继续使用 RSS、GDELT 和 Google News，不会进行无效重试。

---

## 📊 分析报告结构

每次运行后，您会收到包含以下11章的专业分析报告：

### 第1章：重要新闻速览
3-5条最重要的全球事件，今日新闻优先，附历史对比

### 第2章：区域形势分析
- 美洲地区 / 欧洲地区 / 亚洲地区 / 俄罗斯及独联体 / **中东地区**

### 第3章：全球经济趋势分析
宏观指标、**四大央行**货币政策、贸易关系、大宗商品（油/金/铜）

### 第4章：地缘政治风险评估
主要事件、升温信号、经济传导路径

### 第5章：⚡ 黑天鹅事件预警与投资策略

> 对比今日新闻与过去5天历史，识别突发信号

- **5.1** 近期已发生的黑天鹅/灰犀牛事件
- **5.2** 潜在信号识别（军事/金融/政治/能源/科技 五维度）
- **5.3** 黑天鹅风险评级矩阵（含概率、冲击、关注度）
- **5.4** 🎯 **收益最大化投资策略**
  - 事件前预防性布局（A股/美股/港股/日股 各自操作）
  - 事件发生后 T+0~T+3 快速响应清单
  - 潜在收益区间估算
- **5.5** 通用黑天鹅对冲清单（黄金/美元/防御板块比例）

### 第6章：中国A股专项分析
沪深300/科创50走势、行业板块轮动、政策传导、黑天鹅冲击路径、具体股票池建议。新增科技（芯片、AI、存储）、金融和大消费的短期趋势矩阵，统一披露数据日期、催化、失效信号与仓位上限。

### 第7章：美股专项分析
标普500/纳指/道指、Mag-7走势、能源/金融/军工/科技板块、ETF与期权方向建议。科技/半导体与存储、金融、必需及可选消费均以行情、政策和舆情证据进行短期情景分析。

### 第8章：港股专项分析
恒生/恒生科技指数、南下资金、中概科技（腾讯/阿里等）、H股折价逻辑

### 第9章：日本股市专项分析
日经225/TOPIX、日元汇率影响、半导体设备（东京电子等）、日央行加息节奏

### 第10章：未来预测（1-3个月）
四大市场运行路径、黑天鹅概率变化趋势、关键时间节点日历

### 第11章：综合行动建议
- 四大市场仓位分配建议（A股/美股/港股/日股占比）
- 黑天鹅对冲仓位
- 近期关键操作（未来1-2周）
- 风险警戒线

**报告保存位置：**
- JSON格式: `data/reports/report_*.json`
- 文本格式: `data/reports/report_*.txt`
- 邮件: 完整 HTML 渲染（表格/粗体/引用块等）

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

`markets.assets` 可配置美股 Yahoo Finance 标的，以及需要同时提供 `tencent_symbol`、`secid` 的 A 股宽基或 ETF。程序生成 1/5/20 日涨跌、MA5/20/60、量能比和趋势信号；行情源失败时报告会显式降级为新闻情景分析。

`sentiment` 默认仅汇总已抓取的权威新闻标题和市场相对强弱。`guba_enabled` 默认为 `false`；启用股吧标题抓取前，请自行确认数据源条款与使用场景。

`storage.historical_news_days` 和 `storage.historical_news_max_per_day` 分别控制历史新闻窗口及每日输入上限。更长窗口会增加模型上下文与调用成本。

### 调度配置

```yaml
scheduler:
  enabled: true
  run_time: "08:00"       # 每天运行时间（24小时制）
  timezone: "Asia/Shanghai"
```

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
# 查看最新文本报告
ls -lt data/reports/*.txt | head -3

# 预览最新报告
cat $(ls -t data/reports/*.txt | head -1) | head -100
```

### 测试邮件发送

```bash
.venv/bin/python3 src/test_email.py
```

### 清理缓存

系统会自动清理超过配置天数的缓存，也可手动清理：

```bash
rm -rf data/news_cache/*
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
- 检查 LLM 是否正确输出了 Markdown（查看 `data/reports/report_*.txt`）
- 如果 LLM 将整个回复包在 ` ```markdown ``` ` 代码块里，系统会自动剥除
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

### Q5: 没有抓取到新闻？

```bash
# 测试 RSS 源可用性
.venv/bin/python3 src/test_rss_sources.py

# 禁用 News API，只用 RSS
# config.yaml: news_api.enabled: false
```

### Q6: 报告内容质量不满意？

- 更换更强大的模型（如 GPT-4o 或 Claude-3.5-Sonnet）
- 调整 `config.yaml` 的 `analysis.focus_areas`
- 增加 `max_tokens` 以获得更长更详细的分析

---

## 💰 成本估算

### 免费资源
- ✅ RSS Feed（32个源）: 完全免费
- ✅ News API: 免费版100次/天（足够）
- ✅ 邮箱服务: 完全免费
- ✅ 本地 Ollama: 完全免费（需要本地GPU/CPU算力）

### 付费资源（按每次分析估算）
| 模型 | 每次费用 | 月费用（每天1次） |
|------|---------|----------------|
| 本地 Ollama | 免费 | 免费 |
| 讯飞星火 | ~¥0.03-0.06 | ~¥1-2 |
| GPT-4o | ~¥0.3-0.6 | ~¥10-18 |
| Claude 3.5 Sonnet | ~¥0.2-0.4 | ~¥6-12 |

> 报告现在约 11 个完整章节，Token 消耗约 8,000-15,000/次（含历史新闻上下文）。

---

## ⚠️ 注意事项

### 1. 隐私安全
- ⚠️ **不要将 `config.yaml` 提交到公开仓库**（包含 API Key 和邮箱密码）
- 已在 `.gitignore` 中配置忽略

### 2. API 限制
- News API 免费版：100请求/天
- 邮箱：避免频繁发送（建议每天1次）

### 3. 历史新闻缓存
- 历史缓存存于 `data/news_cache/`，供黑天鹅信号对比使用
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

目前系统通过 RSS 获取新闻。如需更实时的黑天鹅预警，可扩展集成：
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
