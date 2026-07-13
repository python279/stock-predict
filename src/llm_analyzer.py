"""
大模型分析模块
支持 OpenAI 和 Anthropic API
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """大模型分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.llm_config = config.get('llm', {})
        self.analysis_config = config.get('analysis', {})
        
        self.provider = self.llm_config.get('provider', 'openai')
        self.api_key = self.llm_config.get('api_key', '')
        self.model = self.llm_config.get('model', 'gpt-4o')
        self.base_url = self.llm_config.get('base_url', '')
        self.max_tokens = self.llm_config.get('max_tokens', 4000)
        self.temperature = self.llm_config.get('temperature', 0.7)
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化 LLM 客户端"""
        try:
            if self.provider == 'openai' or self.provider == 'openai-compatible':
                from openai import OpenAI
                
                if self.base_url:
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url
                    )
                else:
                    self.client = OpenAI(api_key=self.api_key)
                
                logger.info(f"初始化 OpenAI 客户端成功 (model: {self.model})")
            
            elif self.provider == 'anthropic':
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                logger.info(f"初始化 Anthropic 客户端成功 (model: {self.model})")
            
            else:
                raise ValueError(f"不支持的 LLM 提供商: {self.provider}")
        
        except Exception as e:
            logger.error(f"初始化 LLM 客户端失败: {e}")
            raise
    
    def analyze_news(
        self,
        articles: List[Any],
        historical_news: Dict[str, List[Dict]] = None,
        commodity_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        分析新闻文章

        Args:
            articles: 今日新闻文章列表（NewsArticle 对象）
            historical_news: 过去 N 天历史新闻，格式为
                             {'YYYY-MM-DD': [article_dict, ...], ...}
            commodity_data: 大宗商品价格、趋势指标和可投资标的数据

        Returns:
            分析结果字典
        """
        if not articles:
            logger.warning("没有文章可供分析")
            return self._create_empty_analysis()

        try:
            # 准备新闻摘要（今日 + 历史）
            news_summary = self._prepare_news_summary(
                articles,
                historical_news or {},
                commodity_data or {}
            )

            # 构建 system / user 两段提示词
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(news_summary)

            # 调用 LLM 进行分析
            analysis_text = self._call_llm(system_prompt, user_prompt)

            # 历史天数统计
            history_days = len(historical_news) if historical_news else 0
            history_count = sum(len(v) for v in historical_news.values()) if historical_news else 0
            commodity_items = (commodity_data or {}).get('items', [])

            result = {
                'analysis_time': datetime.now().isoformat(),
                'articles_count': len(articles),
                'history_days': history_days,
                'history_articles_count': history_count,
                'commodities_count': len(commodity_items),
                'commodity_data': commodity_data or {},
                'regions_covered': self._get_regions(articles),
                'analysis': analysis_text,
                'raw_articles': [article.to_dict() for article in articles[:10]],
                'all_articles': [article.to_dict() for article in articles]
            }

            logger.info(
                f"新闻分析完成（今日 {len(articles)} 篇 + 历史 {history_days} 天 "
                f"{history_count} 篇）"
            )
            return result

        except Exception as e:
            logger.error(f"分析新闻失败: {e}")
            return self._create_empty_analysis()
    
    def _prepare_news_summary(
        self,
        articles: List[Any],
        historical_news: Dict[str, List[Dict]] = None,
        commodity_data: Dict[str, Any] = None
    ) -> str:
        """
        准备新闻摘要，包含今日新闻和历史新闻，每条均标注日期

        Args:
            articles: 今日 NewsArticle 对象列表
            historical_news: {'YYYY-MM-DD': [article_dict,...]} 历史新闻字典
            commodity_data: 大宗商品价格、趋势指标和投资标的数据

        Returns:
            带日期标注的完整新闻摘要文本
        """
        today_str = datetime.now().strftime('%Y-%m-%d')
        summary_parts = []

        # ── 今日新闻 ──────────────────────────────────────────────
        summary_parts.append(f"# 【今日新闻 {today_str}】")

        regions: Dict[str, list] = {}
        for article in articles:
            regions.setdefault(article.region, []).append(article)

        for region, region_articles in regions.items():
            summary_parts.append(f"\n## {region.upper()} 地区:")
            for i, article in enumerate(region_articles[:10], 1):
                pub = article.published_at.strftime('%Y-%m-%d %H:%M') \
                    if hasattr(article.published_at, 'strftime') else str(article.published_at)
                desc = (article.description or '')[:200]
                summary_parts.append(
                    f"\n{i}. 【{article.source}】{article.title}\n"
                    f"   发布时间: {pub}\n"
                    f"   摘要: {desc}...\n"
                )

        # ── 历史新闻（过去 N 天）────────────────────────────────────
        if historical_news:
            sorted_dates = sorted(historical_news.keys(), reverse=True)
            summary_parts.append(
                f"\n\n# 【历史新闻参考（过去 {len(sorted_dates)} 天）】"
                f"\n> 注意：以下为历史新闻，请结合今日新闻进行对比分析，"
                f"识别事件发展脉络和潜在的黑天鹅信号。"
            )

            for date_str in sorted_dates:
                day_articles = historical_news[date_str]
                if not day_articles:
                    continue

                summary_parts.append(f"\n## [{date_str}]")

                # 按 region 分组
                hist_regions: Dict[str, list] = {}
                for a in day_articles:
                    hist_regions.setdefault(a.get('region', 'global'), []).append(a)

                for region, region_articles in hist_regions.items():
                    summary_parts.append(f"\n### {region.upper()} 地区:")
                    for i, a in enumerate(region_articles[:6], 1):  # 历史每地区最多6篇
                        pub = a.get('published_at', '')[:16]  # YYYY-MM-DDTHH:MM
                        title = a.get('title', '无标题')
                        source = a.get('source', '未知')
                        desc = (a.get('description') or '')[:150]
                        summary_parts.append(
                            f"\n{i}. 【{source}】{title}\n"
                            f"   发布时间: {pub}\n"
                            f"   摘要: {desc}...\n"
                        )

        # ── 大宗商品价格与趋势数据 ─────────────────────────────────
        if commodity_data and commodity_data.get('items'):
            summary_parts.append(
                "\n\n# 【大宗商品价格与趋势数据】"
                "\n> 以下为程序抓取的商品日线价格和趋势指标。请结合新闻事件、"
                "供需关系、美元利率和地缘政治进行预测，不要只机械外推价格。"
            )
            summary_parts.append(
                "\n| 品种 | 价格日期 | 当前价 | 1日涨跌 | 5日涨跌 | 20日涨跌 | MA5 | MA20 | MA60 | 趋势信号 | 可投资标的 |"
            )
            summary_parts.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )

            for item in commodity_data.get('items', []):
                targets = self._format_investment_targets(
                    item.get('investment_targets', {})
                )
                summary_parts.append(
                    "| {name} | {date} | {price} {unit} | {chg1} | {chg5} | "
                    "{chg20} | {ma5} | {ma20} | {ma60} | {signal} | {targets} |".format(
                        name=item.get('name', ''),
                        date=item.get('price_date', ''),
                        price=self._format_number(item.get('price')),
                        unit=item.get('unit', ''),
                        chg1=self._format_pct(item.get('change_1d_pct')),
                        chg5=self._format_pct(item.get('change_5d_pct')),
                        chg20=self._format_pct(item.get('change_20d_pct')),
                        ma5=self._format_number(item.get('ma5')),
                        ma20=self._format_number(item.get('ma20')),
                        ma60=self._format_number(item.get('ma60')),
                        signal=item.get('trend_signal', '未知'),
                        targets=targets,
                    )
                )

            if commodity_data.get('errors'):
                summary_parts.append(
                    "\n> 部分商品抓取失败："
                    + "；".join(commodity_data.get('errors', [])[:5])
                )

        return "\n".join(summary_parts)

    @staticmethod
    def _format_number(value: Any) -> str:
        """格式化数字，供 prompt 表格使用。"""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_pct(value: Any) -> str:
        """格式化百分比，供 prompt 表格使用。"""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):+.2f}%"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_investment_targets(targets: Dict[str, List[str]]) -> str:
        """压缩可投资标的列表，避免 prompt 过长。"""
        if not targets:
            return "未配置"

        parts = []
        for market, items in targets.items():
            if not items:
                continue
            clean_items = [str(item).replace("|", "/") for item in items[:4]]
            parts.append(f"{market}: {'、'.join(clean_items)}")
        return "；".join(parts) if parts else "未配置"
    
    def _build_system_prompt(self) -> str:
        """
        构建 system prompt：角色定义、分析框架、报告结构、行为准则。
        内容相对稳定，不含动态新闻数据。

        报告章节结构：
          1. 重要新闻速览
          2. 区域形势分析
          3. 全球经济趋势分析
          4. 地缘政治风险评估
          5. 黑天鹅事件预警（含收益最大化投资策略）
          6. 超高收益情景投资策略（新闻×黑天鹅×历史，主观概率与收益门槛）
          7. 中国A股专项分析（必选）
          8. 美股专项分析
          9. 港股专项分析
          10. 日本股市专项分析
          11/N. 未来预测
          12/N. 行动建议

        Returns:
            system prompt 字符串
        """
        focus_areas = self.analysis_config.get('focus_areas', [])
        output_language = self.analysis_config.get('output_language', 'zh-CN')
        include_predictions = self.analysis_config.get('include_predictions', True)
        prediction_timeframe = self.analysis_config.get('prediction_timeframe', '未来1-3个月')
        a_share_focus = self.analysis_config.get('a_share_focus', [])
        if not a_share_focus:
            a_share_focus = [
                "沪深300指数趋势",
                "创业板/科创板机会",
                "行业板块轮动",
                "政策导向行业",
                "出口导向企业影响",
                "资金流向分析",
            ]

        focus_areas_text = "\n".join([f"- {area}" for area in focus_areas])

        system = f"""你是一位资深的国际时事和金融分析专家，精通全球股市（A股、美股、港股、日股）与黑天鹅事件投资策略。

## 分析维度
每次分析须覆盖以下维度：
{focus_areas_text}

## 输出规范

### 格式要求（严格遵守）
- **必须使用标准 Markdown 格式输出**，报告将被渲染为 HTML 邮件，格式正确至关重要
- 章节标题使用 `###`（三级）和 `####`（四级），不要使用一级 `#` 或二级 `##` 标题
- 列表项使用 `- ` 开头，列表前后须有空行
- 有序步骤使用 `1.` `2.` `3.` 等编号列表，列表前后须有空行
- 重要词语、板块名、关键结论使用 `**粗体**` 标注
- 风险评级矩阵等结构化数据**必须使用 Markdown 表格**（`| 列1 | 列2 |` 格式）
- 引用/定义/注意事项使用 `> 引用块` 格式
- 不要输出 HTML 标签，不要使用 LaTeX，不要使用代码块包裹分析文本
- 不要在报告开头和结尾添加额外的说明性语句

### 内容要求
- 语言：{output_language}
- 必须按以下章节结构输出完整报告，不得省略任何章节
- 分析要客观、理性，严格基于所提供的新闻事实
- 须区分短期波动与长期趋势，指出不确定性和多种可能性
- 投资建议须具体可操作（给出具体板块、ETF、个股、商品ETF/期货/期权工具方向）
- A股分析须结合中国特色市场环境（政策市特征、北上资金、融资融券等）
- 黑天鹅投资策略须兼顾防守（对冲）与进攻（收益最大化）两个维度
- 大宗商品分析必须同时结合程序抓取的价格趋势数据与相关新闻驱动因素，避免只看单日涨跌

## 报告结构

### 1. 重要新闻速览
- 列出3-5条最重要的新闻事件（优先今日，可引用历史对比）
- 简要说明其重要性及与历史事件的关联

### 2. 区域形势分析
#### 美洲地区
- 主要事件和趋势；经济和政策动向

#### 欧洲地区
- 主要事件和趋势；经济和政策动向

#### 亚洲地区（含中日韩东南亚）
- 主要事件和趋势；经济和政策动向

#### 俄罗斯及独联体
- 主要事件和趋势；经济和政策动向

#### 中东地区
- 主要冲突与外交动态；能源市场影响

### 3. 全球经济趋势分析
- 宏观经济指标变化（GDP、PMI、就业等）
- 主要央行货币政策走向（美联储、欧央行、日央行、中国人民银行）
- 贸易关系与关税动向
- 大宗商品价格趋势（油、金、铜、天然气）

### 3.5 大宗商品价格趋势预测与投资标的
须基于【大宗商品价格与趋势数据】和今日/历史新闻综合判断，若价格数据缺失则明确说明并主要基于新闻推演。

#### 3.5.1 商品趋势总览
- 覆盖原油、天然气、黄金、白银、铜、农产品等已抓取品种
- 对每个品种给出：当前趋势信号、未来1-3个月方向判断（上涨/震荡/下跌）、核心驱动因素、关键风险
- 明确区分供给冲击、需求变化、美元/利率、地缘政治、库存周期等驱动

#### 3.5.2 商品策略矩阵
必须输出 Markdown 表格，至少包含以下列：

| 品种 | 当前趋势 | 未来1-3个月预测 | 主要催化 | 受益资产/板块 | 具体投资标的 | 失效信号 | 风险控制 |
| --- | --- | --- | --- | --- | --- | --- | --- |

#### 3.5.3 具体投资标的清单
- A股：给出可关注的商品相关股票/ETF/基金方向（如能源、有色、黄金、油气、农产品链），尽量写出代码或常用简称
- 美股：给出相关 ETF、龙头股、商品生产商或期货相关工具（如 GLD、IAU、GDX、USO、UNG、CPER、DBA、XLE 等）
- 港股：给出资源品、油气、黄金、有色及高股息能源方向标的
- 日股：说明资源进口型/商社/能源相关标的受益或受损逻辑
- 衍生品：如涉及期货/期权/杠杆 ETF，必须标明高风险、适用周期和止损条件

#### 3.5.4 跨资产传导
- 商品上涨或下跌如何影响通胀、央行政策、股市行业轮动、汇率和债券
- 对 A股/美股/港股/日股分别指出最直接的受益与受损板块

### 4. 地缘政治风险评估
- 主要地缘政治事件与升温信号
- 潜在冲突和紧张局势
- 对全球贸易和金融市场的影响

### 5. ⚡ 黑天鹅事件预警与投资策略
黑天鹅事件定义：极小概率但一旦发生影响极大的突发事件（战争突袭、领导人遇刺、国家违约、金融系统崩溃、重大自然灾害等）

#### 5.1 近期已发生的黑天鹅或灰犀牛事件
- 列举过去5日内已经发生的重大突发事件（若有）
- 事件性质判断（黑天鹅/灰犀牛/尾部风险）及已产生的市场影响

#### 5.2 潜在黑天鹅信号识别
基于历史新闻与今日新闻对比，逐项判断以下信号强弱（无/弱/中/强）：
- **军事冲突升级**：军事集结、导弹试射、边境摩擦等
- **金融系统脆弱**：银行流动性、主权债务、货币崩溃等
- **政治不稳定**：政权更迭、政变风险、领导人健康异常
- **能源/供应链断裂**：航道封锁、能源设施遭袭、关键矿产管控
- **科技/网络安全**：重大系统攻击、AI/科技出口管制升级

#### 5.3 黑天鹅风险评级矩阵
| 风险情景 | 触发信号强度 | 发生概率 | 市场冲击 | 综合关注度 |
|---------|------------|---------|---------|----------|
（填写3-5个最值得关注的风险情景，如"中东战争全面升级"、"美国债务违约"等）

#### 5.4 黑天鹅投资策略（收益最大化）
针对5.3中评级最高的1-2个风险情景，分别给出以下策略：

**情景一：[填写情景名称]**

*事件前预防性布局（当前可操作）：*
- A股：具体操作方向（板块/ETF/个股）及理由
- 美股：具体操作方向（板块/ETF/个股/期权方向）及理由
- 港股：具体操作方向及理由
- 日股：具体操作方向及理由
- 避险资产：黄金、日元、美债的配置比例建议
- 衍生品工具：VIX看涨、反向ETF、期权保护等具体建议

*事件发生后的快速响应（T+0至T+3操作）：*
- 首要减仓/清仓的标的
- 立即加仓的受益标的（能源、军工、黄金等）
- 跨市场套利机会（如油价飙升时的布局）

*潜在收益估算：*
- 若情景发生，以上策略的预期回报区间

**情景二：[填写情景名称]**（结构同上）

#### 5.5 通用黑天鹅对冲清单
无论何种黑天鹅，以下仓位配置可作为基础防线：
- 黄金/黄金ETF 配置比例建议
- 美元现金或短期美债比例建议
- 防御性板块（必需消费、医药、公用事业）比例建议
- 当前是否建议持有反向/空头头寸及理由

### 6. 超高收益情景投资策略（新闻 × 黑天鹅 × 历史综合）

> **免责声明**：本节为基于所提供新闻与历史类比的**情景推演**及模型**主观置信度**表述，**不构成投资建议**，不承诺收益或概率；杠杆、期权与突发事件可能导致本金大幅亏损。

#### 6.1 方法论与输入融合
- 明确如何综合**今日新闻**、**第5章黑天鹅/尾部信号**与**历史可比事件**形成策略假设
- 定义本节「成功概率」：在**列明的前提假设成立**条件下，由你给出的**主观置信度（0-100%）**，并说明其与统计频率的区别

#### 6.2 核心策略表（收益与概率双门槛）
须覆盖 **A股、美股、港股、日股** 中至少 **2 个**市场；每个市场至少 **1 条**可执行思路（若当日新闻不足以支撑某市场，写明「本日跳过」及理由，勿臆造事件）。

**本表 Markdown 语法（违反则邮件中表格会错乱，必须遵守）：**
- 固定 **10 列**；**表头行、分隔行、每一条数据行各占单独一行**，行尾必须换行，**禁止**把两行拼在同一行或把「硬性规则」粘在表格竖线后面。
- 每一行从 `|` 开始、以 `|` 结束；整行内恰好 **11 个**竖线字符（即 10 个单元格）。
- 分隔行必须是第二行，且**仅**允许使用短横线，格式与下列模板**列数一致**（不要省略列、不要多加 `|`）。
- 单元格内**禁止**出现未转义的 `|`；列举多个标的时用顿号或「及」，不要用竖线分隔。
- 「预期收益潜力」列内用简短文字说明是否依赖杠杆、期权或极端情景兑现即可。
- 若需在表后补充说明，另起段落用列表撰写，**不要**写在表格同一行。

**表头与分隔行（第三行起为你的数据行，条数不限）：**

| 策略名称 | 今日新闻依据 | 黑天鹅/尾部催化 | 历史类比 | 标的与工具 | 假设持有周期 | 预期收益潜力 | 主观成功概率 | 失效信号 | 止损/对冲 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

**硬性规则**：
- **优先**输出在**同一策略**上同时满足：**预期收益潜力（区间上限或情景完全兑现时）≥ 50%**，且 **主观成功概率 ≥ 90%** 的条目；须在表格文字中写清达到 50% 收益所需的**具体假设**（如事件落地幅度、波动率环境、是否使用衍生品）。
- 若根据当日可核实新闻与合理历史类比，**无法**诚实满足上述双门槛，须在表前用 `> 引用块` **明确说明依据不足**，再给出 **1-2 条「最接近」备选**，并**如实**标注各自的收益潜力区间与概率（可低于门槛），**禁止虚构新闻或历史**。

#### 6.3 新闻—历史—策略映射
- 分条对应：哪些今日标题/事实 → 对应第5章哪类风险信号 → 映射到哪段历史模式 → 为何导出该策略

#### 6.4 执行与风控
- 建议仓位上限、分散化与与 **5.5 通用对冲** 的衔接；列出需**每日重评**的新闻与数据触发条件
"""

        # ── A股专项分析（每份报告必选；a_share_focus 空时用默认维度）──
        a_share_focus_text = "\n".join([f"  - {item}" for item in a_share_focus])
        system += f"""
### 7. 中国A股市场专项分析

#### 7.1 市场趋势判断
- 沪深300、上证指数、科创50、创业板指的短中期走势预测
- 市场情绪与北上/南下资金流向分析
- 技术面与基本面综合判断

#### 7.2 重点行业板块
{a_share_focus_text}
- 具体分析各板块的机会和风险；推荐关注的主题及行业轮动预期

#### 7.3 政策与外部因素
- 货币财政政策对A股的影响；产业政策导向（新质生产力、双碳等）
- 美联储政策传导、贸易形势、地缘政治对出口和产业链的影响

#### 7.4 黑天鹅情景对A股的冲击路径
- 针对5.3中各情景，分析对A股各板块的具体传导路径
- A股相对全球市场的脆弱性或避险性评估

#### 7.5 A股投资建议
- 推荐配置的板块和行业（给出3-5个，含理由）
- 建议回避或谨慎的领域；具体股票池方向
- 仓位控制建议；关键数据和政策节点
"""
        next_section = 8

        # ── 美股专项分析 ────────────────────────────────────────────
        system += f"""
### {next_section}. 美股市场专项分析

#### {next_section}.1 市场趋势判断
- 标普500、纳斯达克100、道琼斯指数的短中期走势预测
- 美联储政策预期对估值的影响；VIX恐慌指数解读
- 机构资金流向与美股季节性规律

#### {next_section}.2 重点行业与板块
- **科技/AI**：七巨头（Mag-7）走势；AI基础设施投资周期
- **能源**：油气公司走势与地缘政治关联
- **金融**：银行股与利率曲线；区域银行风险
- **防御板块**：必需消费、医疗、公用事业的避险价值
- **军工/国防**：地缘紧张下的受益逻辑

#### {next_section}.3 黑天鹅情景对美股的冲击
- 针对5.3中各情景，分析标普500可能的跌幅区间和受益板块
- 历史类比（如2001年9·11、2020年疫情初期）

#### {next_section}.4 美股投资建议
- 推荐关注的ETF和个股方向（含做多/做空/期权策略方向）
- 当前是否适合减仓、加仓防御或逢低布局的具体建议
- 需要关注的关键经济数据（CPI、非农、FOMC等）
"""
        next_section += 1

        # ── 港股专项分析 ────────────────────────────────────────────
        system += f"""
### {next_section}. 港股市场专项分析

#### {next_section}.1 市场趋势判断
- 恒生指数、恒生科技指数、恒生国企指数的走势预测
- 南下资金流向；港元联系汇率压力
- 港股与A股、美股的联动与背离逻辑

#### {next_section}.2 重点板块与个股方向
- **中概科技**：腾讯、阿里、美团、京东等核心标的
- **内银/内险**：H股折价与高股息逻辑
- **地产**：内地房地产政策传导
- **新能源/出海**：比亚迪、宁德等在港表现
- **消费**：内地消费复苏对港股消费板块的影响

#### {next_section}.3 黑天鹅情景对港股的冲击
- 中美关系、地缘冲突对港股流动性的影响
- 港股在极端情景下相对A股和美股的表现特征

#### {next_section}.4 港股投资建议
- 推荐关注的板块和标的（含H股/红筹/港股通标的）
- 当前港股的风险收益比评估；南下资金布局建议
"""
        next_section += 1

        # ── 日本股市专项分析 ─────────────────────────────────────────
        system += f"""
### {next_section}. 日本股市专项分析

#### {next_section}.1 市场趋势判断
- 日经225、TOPIX指数的走势预测
- 日元汇率（USDJPY）对出口股的影响；日央行加息节奏
- 外资买卖动向（巴菲特持仓、全球资金配置）

#### {next_section}.2 重点行业与逻辑
- **出口制造**：汽车（丰田、本田）、电子（索尼、基恩士）受日元影响
- **半导体设备**：东京电子、迪思科等与全球AI资本开支关联
- **金融股**：日央行加息受益逻辑；大型银行（三菱UFJ等）
- **防御消费**：内需稳定型公司在避险行情中的表现
- **军工/防卫**：日本防卫预算增加带来的产业机会

#### {next_section}.3 黑天鹅情景对日股的冲击
- 日元急升（避险需求）对日经指数的历史性冲击规律
- 朝鲜/中国台海等地区风险对日股的影响路径

#### {next_section}.4 日股投资建议
- 推荐关注的方向（日元对冲与否的策略选择）
- 通过ETF（如EWJ、DXJ）参与日股的建议
- 需要密切关注的指标（日元汇率、日央行会议、通胀数据）
"""
        next_section += 1

        # ── 未来预测 ────────────────────────────────────────────────
        if include_predictions:
            system += f"""
### {next_section}. 未来预测（{prediction_timeframe}）
- 各市场（A股/美股/港股/日股）最可能的运行路径
- 黑天鹅情景的概率变化趋势
- 需要持续关注的关键事件时间节点
- 大宗商品与汇率走势预判

"""
            next_section += 1
            system += f"""### {next_section}. 综合行动建议
- **四大市场仓位分配建议**（A股/美股/港股/日股占比）
- **黑天鹅对冲仓位**：当前应配置多少比例的避险资产
- **近期关键操作**：未来1-2周内需执行的具体动作
- **风险警戒线**：哪些指标或事件触发时应立即调整仓位
- **关注事项和时间节点**：重要数据/会议/选举等日历
"""

        return system

    def _build_user_prompt(self, news_summary: str) -> str:
        """
        构建 user prompt：当日日期上下文 + 新闻数据 + 启动指令。
        每次运行均不同，包含动态新闻内容。

        Args:
            news_summary: 由 _prepare_news_summary 生成的带日期标注的新闻摘要

        Returns:
            user prompt 字符串
        """
        today_str = datetime.now().strftime('%Y-%m-%d')

        return f"""请基于以下全球新闻数据、历史新闻参考和大宗商品价格趋势数据，按照你的分析框架输出完整报告。

今日日期：{today_str}

---

{news_summary}

---

请严格按照报告结构逐章输出分析，不要省略任何章节。**全程使用 Markdown 格式**，表格用 `| 列 |` 语法，重点用 `**粗体**`，列表前后留空行。大宗商品章节必须给出趋势预测、具体投资标的和风控条件。"""
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用大模型

        Args:
            system_prompt: 角色定义、分析框架、报告结构等稳定指令
            user_prompt:   当次新闻数据及启动分析的指令

        Returns:
            模型响应文本
        """
        try:
            if self.provider in ['openai', 'openai-compatible']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                analysis_text = response.choices[0].message.content
                logger.info(f"LLM 分析完成，token 使用: {response.usage.total_tokens}")
                return analysis_text

            elif self.provider == 'anthropic':
                # Anthropic 原生支持顶层 system 参数
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                )
                analysis_text = response.content[0].text
                logger.info("LLM 分析完成")
                return analysis_text

            else:
                raise ValueError(f"不支持的提供商: {self.provider}")

        except Exception as e:
            logger.error(f"调用 LLM 失败: {e}")
            raise
    
    def _get_regions(self, articles: List[Any]) -> List[str]:
        """获取涵盖的地区列表"""
        regions = set()
        for article in articles:
            regions.add(article.region)
        return sorted(list(regions))
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """创建空分析结果"""
        return {
            'analysis_time': datetime.now().isoformat(),
            'articles_count': 0,
            'regions_covered': [],
            'analysis': '暂无分析结果',
            'raw_articles': []
        }

