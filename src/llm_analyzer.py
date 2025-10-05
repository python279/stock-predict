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
    
    def analyze_news(self, articles: List[Any]) -> Dict[str, Any]:
        """
        分析新闻文章
        
        Args:
            articles: 文章列表
        
        Returns:
            分析结果字典
        """
        if not articles:
            logger.warning("没有文章可供分析")
            return self._create_empty_analysis()
        
        try:
            # 准备新闻摘要
            news_summary = self._prepare_news_summary(articles)
            
            # 构建分析提示词
            prompt = self._build_analysis_prompt(news_summary)
            
            # 调用 LLM 进行分析
            analysis_text = self._call_llm(prompt)
            
            # 解析分析结果
            result = {
                'analysis_time': datetime.now().isoformat(),
                'articles_count': len(articles),
                'regions_covered': self._get_regions(articles),
                'analysis': analysis_text,
                'raw_articles': [article.to_dict() for article in articles[:10]],  # 保存前10篇原始文章
                'all_articles': [article.to_dict() for article in articles]  # 保存所有文章用于引用
            }
            
            logger.info("新闻分析完成")
            return result
        
        except Exception as e:
            logger.error(f"分析新闻失败: {e}")
            return self._create_empty_analysis()
    
    def _prepare_news_summary(self, articles: List[Any]) -> str:
        """
        准备新闻摘要
        
        Args:
            articles: 文章列表
        
        Returns:
            新闻摘要文本
        """
        summary_parts = []
        
        # 按区域分组
        regions = {}
        for article in articles:
            region = article.region
            if region not in regions:
                regions[region] = []
            regions[region].append(article)
        
        # 生成摘要
        for region, region_articles in regions.items():
            summary_parts.append(f"\n## {region.upper()} 地区新闻:")
            
            for i, article in enumerate(region_articles[:10], 1):  # 每个地区最多10篇
                summary_parts.append(
                    f"\n{i}. 【{article.source}】{article.title}\n"
                    f"   时间: {article.published_at.strftime('%Y-%m-%d %H:%M')}\n"
                    f"   摘要: {article.description[:200]}...\n"
                )
        
        return "\n".join(summary_parts)
    
    def _build_analysis_prompt(self, news_summary: str) -> str:
        """
        构建分析提示词
        
        Args:
            news_summary: 新闻摘要
        
        Returns:
            提示词
        """
        focus_areas = self.analysis_config.get('focus_areas', [])
        output_language = self.analysis_config.get('output_language', 'zh-CN')
        include_predictions = self.analysis_config.get('include_predictions', True)
        prediction_timeframe = self.analysis_config.get('prediction_timeframe', '未来1-3个月')
        
        focus_areas_text = "\n".join([f"- {area}" for area in focus_areas])
        
        prompt = f"""你是一位资深的国际时事和金融分析专家。请基于以下全球新闻，进行深度分析。

# 今日全球新闻摘要

{news_summary}

# 分析要求

请从以下维度进行分析：
{focus_areas_text}

## 分析结构

请按照以下结构输出分析报告（使用{output_language}）：

### 1. 重要新闻速览
- 列出3-5条最重要的新闻事件
- 简要说明其重要性

### 2. 区域形势分析
#### 美洲地区
- 主要事件和趋势
- 经济和政策动向

#### 欧洲地区
- 主要事件和趋势
- 经济和政策动向

#### 亚洲地区
- 主要事件和趋势
- 经济和政策动向

#### 俄罗斯及独联体
- 主要事件和趋势
- 经济和政策动向

### 3. 全球经济趋势分析
- 宏观经济指标变化
- 货币政策走向
- 贸易关系变化
- 大宗商品价格趋势

### 4. 地缘政治风险评估
- 主要地缘政治事件
- 潜在冲突和紧张局势
- 对全球经济的影响

### 5. 股市影响分析
- 对全球股市的影响
- 具体板块机会和风险
- 关注的上市公司或行业

### 6. 未来预测（{prediction_timeframe}）
"""
        
        if include_predictions:
            prompt += """- 可能发生的重要事件
- 经济趋势预测
- 投资建议和风险警示
- 需要关注的关键指标

### 7. 行动建议
- 投资组合调整建议
- 风险对冲策略
- 关注事项和时间节点
"""
        
        prompt += """

## 注意事项
- 分析要客观、理性，基于事实
- 指出不确定性和多种可能性
- 区分短期波动和长期趋势
- 提供可操作的建议

请开始你的分析："""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用大模型
        
        Args:
            prompt: 提示词
        
        Returns:
            模型响应
        """
        try:
            if self.provider in ['openai', 'openai-compatible']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一位专业的国际时事和金融分析专家，擅长分析全球政治经济形势，预测市场趋势。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                analysis_text = response.choices[0].message.content
                logger.info(f"LLM 分析完成，token 使用: {response.usage.total_tokens}")
                return analysis_text
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
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

