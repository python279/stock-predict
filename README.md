# 全球新闻自动分析系统

## 📋 项目简介

这是一个完全自动化的国际新闻抓取、分析和通知系统。系统每天自动抓取世界各地的权威新闻源（美国、欧洲、东南亚、俄罗斯等），使用大模型（OpenAI/Anthropic/讯飞星火）进行深度分析，预测世界局势和股市影响，并通过邮件自动发送分析报告。

## ✨ 主要特性

- 🌍 **多源新闻抓取**: News API + 19个权威RSS新闻源（100%可用）
- 🤖 **智能AI分析**: 使用大模型深度分析全球政治经济形势
- 📊 **股市预测**: 分析新闻对股市的影响并提供专业投资建议
- 📧 **自动邮件**: 每天定时发送精美的HTML格式分析报告
- ⏰ **自动调度**: 支持定时自动执行，完全无人值守
- 🛡️ **高可靠性**: 完善的错误处理和重试机制
- ⚙️ **高度可配置**: 所有参数均可通过配置文件调整
- 📦 **模块化设计**: 清晰的代码结构，易于扩展和维护

## 🏗️ 项目结构

```
info-os/
├── config.yaml              # 主配置文件
├── requirements.txt         # Python依赖
├── README.md               # 本文件
├── run.sh                  # 快速启动脚本
├── src/                    # 源代码目录
│   ├── config_loader.py    # 配置加载模块
│   ├── news_fetcher.py     # 新闻抓取模块（News API + RSS）
│   ├── llm_analyzer.py     # 大模型分析模块
│   ├── email_sender.py     # 邮件发送模块
│   ├── data_storage.py     # 数据存储模块
│   ├── logger_config.py    # 日志配置模块
│   ├── main.py            # 主程序
│   ├── scheduler.py       # 定时调度器
│   ├── test_config.py     # 配置测试工具
│   ├── test_rss_sources.py # RSS源测试工具
│   └── test_email.py      # 邮件测试工具
├── data/                   # 数据目录
│   ├── news_cache/        # 新闻缓存
│   └── reports/           # 分析报告（JSON + TXT）
└── logs/                   # 日志目录
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
cd /Users/lhq/Develop/info-os

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
  recipient_email: "your-qq-number@qq.com"
```

#### 选项B: 使用Gmail

1. **启用两步验证**
   - 访问: https://myaccount.google.com/security
   - 启用"两步验证"

2. **生成应用专用密码**
   - 访问: https://myaccount.google.com/apppasswords
   - 选择应用: "邮件"
   - 选择设备: "Mac电脑"或"其他"
   - 复制生成的16位密码

3. **更新配置文件**：
```yaml
email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "your-email@gmail.com"
  sender_password: "xxxx xxxx xxxx xxxx"  # 应用专用密码
  recipient_email: "your-email@gmail.com"
```

**注意**: 
- Gmail需要使用应用专用密码，不是登录密码
- 如果连接Gmail超时，建议切换到QQ邮箱或163邮箱

#### 其他邮箱选项

**163邮箱:**
```yaml
email:
  smtp_server: "smtp.163.com"
  smtp_port: 465
  sender_email: "your-email@163.com"
  sender_password: "授权码"
```

**126邮箱:**
```yaml
email:
  smtp_server: "smtp.126.com"
  smtp_port: 465
  sender_email: "your-email@126.com"
  sender_password: "授权码"
```

### 步骤 3: 配置大模型API

系统支持多种大模型，选择其中一种配置即可。

#### OpenAI（GPT-4）
```yaml
llm:
  provider: "openai"
  api_key: "sk-..."
  model: "gpt-4o"
  base_url: ""  # 留空使用官方API
  max_tokens: 4000
  temperature: 0.7
```

#### OpenAI兼容接口（讯飞星火、通义千问等）
```yaml
llm:
  provider: "openai-compatible"
  api_key: "your-api-key"
  model: "x1"  # 或其他模型名称
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
  max_tokens: 4000
  temperature: 0.7
```

### 步骤 4: 配置News API（可选）

访问 https://newsapi.org 注册获取免费API Key（100次/天）：

```yaml
news_api:
  enabled: true
  api_key: "your-newsapi-key"
```

**不配置也可以**：系统会自动使用19个RSS新闻源，完全够用。

### 步骤 5: 测试配置

运行测试脚本验证配置：

```bash
.venv/bin/python3 src/test_config.py
```

如果看到所有测试通过，说明配置成功！

### 步骤 6: 立即运行

**立即执行一次（推荐先试试）：**
```bash
.venv/bin/python3 src/main.py
```

系统会：
1. 抓取全球新闻（约50秒）
2. AI分析（约49秒）
3. 发送邮件报告

总耗时约2分钟。

**启动定时调度器（每天自动运行）：**
```bash
.venv/bin/python3 src/scheduler.py
```

系统会每天8:00自动运行（可在config.yaml中修改时间）。

**使用启动脚本（最简单）：**
```bash
./run.sh
```

选择运行模式即可。

## 📰 新闻源配置（19个，100%可用）

系统经过严格测试，配置了19个权威新闻源，**测试通过率100%**。

### 全球综合（3个）
1. ✅ **BBC World News** - 英国公共广播，32篇新闻
2. ✅ **BBC Business** - BBC商业频道，40篇新闻
3. ✅ **Al Jazeera** - 半岛电视台，50篇新闻

### 美洲（3个）
4. ✅ **NPR News** - 美国公共广播，10篇新闻
5. ✅ **CNN Top Stories** - CNN头条新闻
6. ✅ **CNN World** - CNN世界新闻

### 欧洲（4个）
7. ✅ **Financial Times** - 英国金融时报，9篇新闻
8. ✅ **Deutsche Welle** - 德国之声，151篇新闻
9. ✅ **France 24** - 法国国际新闻，24篇新闻
10. ✅ **The Guardian World** - 卫报世界版

### 亚洲和东南亚（5个）
11. ✅ **Channel NewsAsia** - 新加坡亚洲新闻台，20篇新闻
12. ✅ **Nikkei Asia** - 日经亚洲，50篇新闻
13. ✅ **The Straits Times** - 新加坡海峡时报，99篇新闻
14. ✅ **Bangkok Post** - 曼谷邮报，10篇新闻
15. ✅ **South China Morning Post** - 南华早报

### 俄罗斯（2个）
16. ✅ **TASS** - 俄罗斯塔斯社，45篇新闻
17. ✅ **Moscow Times** - 莫斯科时报，50篇新闻

### 国际组织（2个）
18. ✅ **UN News** - 联合国新闻，30篇新闻
19. ✅ **IMF News** - 国际货币基金组织，10篇新闻

### 新闻源特点

**地域覆盖均衡：**
- 全球综合: 3个源
- 美洲: 3个源
- 欧洲: 4个源
- 亚洲: 5个源
- 俄罗斯: 2个源
- 国际组织: 2个源

**媒体类型多样：**
- 公共广播: BBC, NPR, Deutsche Welle, France 24
- 商业媒体: CNN, Financial Times, The Guardian, SCMP
- 通讯社: TASS
- 国际组织: UN, IMF

**观点平衡：**
- 西方视角: BBC, CNN, The Guardian, Financial Times
- 中东视角: Al Jazeera
- 亚洲视角: Nikkei Asia, SCMP, Channel NewsAsia
- 俄罗斯视角: TASS, Moscow Times
- 国际组织视角: UN, IMF

### 新闻获取能力

- **RSS Feed**: 19个源，每次约200篇新闻
- **News API**: 10国家×3类别×10篇 = 约300篇新闻
- **总计**: 每次可抓取约**500篇全球新闻**
- **去重过滤后**: 保留最重要的50篇进行AI分析

### 测试RSS源

可以随时测试RSS源的可用性：
```bash
.venv/bin/python3 src/test_rss_sources.py
```

## 📊 分析报告示例

每次运行后，您会收到包含以下内容的专业分析报告：

### 1. 重要新闻速览
3-5条最重要的全球事件，附重要性说明

### 2. 区域形势分析
- **美洲地区**: 主要事件、经济与政策动向
- **欧洲地区**: 主要事件、经济与政策动向
- **亚洲地区**: 主要事件、经济与政策动向
- **俄罗斯及独联体**: 主要事件、经济与政策动向

### 3. 全球经济趋势分析
- 宏观经济指标变化
- 货币政策走向
- 贸易关系变化
- 大宗商品价格趋势

### 4. 地缘政治风险评估
- 核心风险点
- 潜在冲突领域
- 经济传导路径

### 5. 股市影响分析
- 整体市场影响
- 机会板块（哪些板块受益）
- 风险警示（哪些板块受损）
- 关注的上市公司或行业

### 6. 未来预测（1-3个月）
- 关键事件预判
- 经济趋势预测
- 投资建议
- 需要监测的关键指标

### 7. 行动建议
- 投资组合调整建议
- 风险对冲策略
- 关注事项和时间节点

**报告保存位置：**
- JSON格式: `data/reports/report_*.json`
- 文本格式: `data/reports/report_*.txt`
- 邮件: HTML精美格式

## ⚙️ 高级配置

### 调度配置

```yaml
scheduler:
  enabled: true
  run_time: "08:00"  # 每天运行时间（24小时制）
  timezone: "Asia/Shanghai"  # 时区
```

### 内容过滤

```yaml
content_filter:
  min_content_length: 100  # 最小内容长度
  exclude_keywords:  # 排除包含这些关键词的文章
    - "horoscope"
    - "celebrity gossip"
    - "entertainment"
    - "sports"
  include_keywords:  # 优先包含这些关键词的文章
    - "economy"
    - "market"
    - "trade"
    - "gdp"
    - "inflation"
```

### 分析配置

```yaml
analysis:
  focus_areas:  # 分析重点
    - "全球经济趋势"
    - "地缘政治风险"
    - "股市影响因素"
    - "贸易关系变化"
  output_language: "zh-CN"  # 输出语言
  include_predictions: true  # 是否包含预测
  prediction_timeframe: "未来1-3个月"  # 预测时间范围
```

### 添加新的新闻源

编辑 `config.yaml`，在 `rss_feeds.sources` 中添加：

```yaml
- name: "新的新闻源"
  url: "https://example.com/rss"
  region: "asia"  # global, americas, europe, asia, russia, international
  priority: 4  # 1-5，数字越大优先级越高
```

## 🛠️ 运维管理

### 查看日志

```bash
# 实时查看日志
tail -f logs/news_analyzer.log

# 查看最近100行
tail -100 logs/news_analyzer.log
```

### 查看报告

```bash
# 查看最新的文本报告
cat data/reports/report_*.txt | tail -200

# 查看JSON报告
cat data/reports/report_*.json | jq .
```

### 测试邮件发送

```bash
.venv/bin/python3 src/test_email.py
```

### 清理缓存

系统会自动清理7天前的缓存，也可以手动清理：

```bash
rm -rf data/news_cache/*
```

### 使用Cron定时运行

如果不使用内置调度器，可以使用系统Cron：

```bash
crontab -e
```

添加以下行（每天8:00运行）：
```
0 8 * * * cd /Users/lhq/Develop/info-os/src && /usr/bin/python3 main.py >> /Users/lhq/Develop/info-os/logs/cron.log 2>&1
```

### 后台运行调度器

```bash
nohup .venv/bin/python3 src/scheduler.py > logs/scheduler.log 2>&1 &

# 查看进程
ps aux | grep scheduler

# 停止
kill <PID>
```

## 🔍 常见问题

### Q1: 邮件发送失败？

**Gmail超时**: 
- 网络无法访问Gmail SMTP
- 需要配置代理或切换到QQ邮箱

**认证失败**:
- 确保使用授权码/应用专用密码，不是登录密码
- Gmail需要先启用两步验证

**解决方案**:
```bash
# 测试邮件
.venv/bin/python3 src/test_email.py

# 推荐切换到QQ邮箱（国内稳定）
```

### Q2: 大模型API调用失败？

检查：
- API Key是否正确
- 是否有足够配额
- base_url是否正确（OpenAI兼容接口）
- 网络连接是否正常

### Q3: 没有抓取到新闻？

可能原因：
- 网络连接问题
- 某些国外RSS需要代理
- News API配额用完

解决方案：
```bash
# 测试RSS源
.venv/bin/python3 src/test_rss_sources.py

# 禁用News API，只用RSS
# 在config.yaml中设置: news_api.enabled: false
```

### Q4: 报告质量不满意？

调整分析配置：
```yaml
analysis:
  focus_areas:
    - "你关心的维度1"
    - "你关心的维度2"
```

或更换更强大的大模型（如GPT-4）。

### Q5: 如何查看错误详情？

```bash
# 查看日志
tail -100 logs/news_analyzer.log

# 运行配置测试
.venv/bin/python3 src/test_config.py
```

## 💰 成本估算

### 免费资源
- ✅ RSS Feed: 完全免费
- ✅ News API: 免费版100次/天（足够使用）
- ✅ 邮箱服务: 完全免费（QQ、Gmail、163等）

### 付费资源
- 大模型API（根据选择）:
  - OpenAI GPT-4: 约¥0.2-0.4/次
  - 讯飞星火: 约¥0.03-0.06/次
  - Claude: 约¥0.15-0.3/次

**预计月成本**: ¥1-12（每天运行一次，取决于选择的大模型）

推荐使用讯飞星火等国内模型，成本更低且速度快。

## 📈 性能指标

- **新闻抓取速度**: 约50秒（19个RSS + News API）
- **AI分析速度**: 约49秒（讯飞星火x1）
- **总耗时**: 约2分钟
- **成功率**: 100%（RSS源全部可用）
- **Token消耗**: 约5,000-6,000 tokens/次

## ⚠️ 注意事项

### 1. 隐私安全
- ⚠️ **不要将 `config.yaml` 提交到公开仓库**
- API Key和密码都是敏感信息
- 已在 `.gitignore` 中配置忽略

### 2. API限制
- News API免费版: 100请求/天
- 大模型API: 根据你的套餐限制
- 邮箱: 避免频繁发送（建议每天1次）

### 3. 新闻源可靠性
- 系统已配置权威媒体，排除了不可靠来源
- 建议定期测试RSS源可用性
- 如果某个源失效，及时替换

### 4. 法律合规
- 仅用于个人学习和研究
- 遵守各新闻源的使用条款
- 不用于商业目的或转发

## 🤝 扩展开发

### 添加新的通知方式

可以扩展 `email_sender.py`，添加：
- Slack通知
- 微信通知（企业微信webhook）
- Telegram Bot
- 钉钉通知

### 自定义分析维度

修改 `config.yaml`:
```yaml
analysis:
  focus_areas:
    - "加密货币市场"
    - "AI行业发展"
    - "新能源汽车"
    - "你关心的任何维度"
```

### 更换大模型

修改 `llm_analyzer.py`，支持新的API接口。

## 📝 维护建议

### 定期检查（每月一次）

1. **测试RSS源状态**
```bash
.venv/bin/python3 src/test_rss_sources.py
```

2. **检查API配额**
- News API使用量
- 大模型API使用量

3. **审查报告质量**
- 查看最近的分析报告
- 根据需要调整配置

4. **查看日志异常**
```bash
grep ERROR logs/news_analyzer.log
```

### 备份重要数据

```bash
# 备份配置
cp config.yaml config.yaml.backup

# 备份报告（可选）
tar -czf reports_backup_$(date +%Y%m%d).tar.gz data/reports/
```

## 📄 许可证

MIT License - 可自由使用、修改和分发

## 💬 联系方式

如有问题或建议：
- Email: haiqiang2linux@gmail.com
- 查看日志: `logs/news_analyzer.log`
- 运行测试: `python src/test_config.py`

---

## 🎉 开始使用

现在你已经了解了所有内容，可以：

1. ✅ 创建虚拟环境: `python3 -m venv .venv`
2. ✅ 安装依赖: `.venv/bin/pip install -r requirements.txt`
3. ✅ 配置邮箱: 编辑 `config.yaml`（推荐QQ邮箱）
4. ✅ 配置大模型: 填入你的API Key
5. ✅ 测试配置: `.venv/bin/python3 src/test_config.py`
6. ✅ 立即运行: `.venv/bin/python3 src/main.py`
7. ✅ 启动调度: `.venv/bin/python3 src/scheduler.py`

**⚡️ 让AI帮你每天洞察世界，把握投资机会！**
