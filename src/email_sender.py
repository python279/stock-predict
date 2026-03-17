"""
邮件发送模块
支持通过 Gmail SMTP 发送邮件
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import markdown as md_lib

logger = logging.getLogger(__name__)


class EmailSender:
    """邮件发送器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化邮件发送器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.email_config = config.get('email', {})
        
        self.smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.email_config.get('smtp_port', 587)
        self.sender_email = self.email_config.get('sender_email', '')
        self.sender_password = self.email_config.get('sender_password', '')
        
        # 支持多个收件人（向后兼容）
        recipient_emails = self.email_config.get('recipient_emails', [])
        recipient_email = self.email_config.get('recipient_email', '')
        
        if recipient_emails:
            # 使用新的 recipient_emails 列表
            self.recipient_emails = recipient_emails
        elif recipient_email:
            # 向后兼容：将单个邮箱转换为列表
            self.recipient_emails = [recipient_email]
        else:
            self.recipient_emails = []
        
        logger.info(f"邮件发送器初始化完成，收件人数量: {len(self.recipient_emails)}")
    
    def send_analysis_report(
        self,
        analysis_result: Dict[str, Any],
        subject_prefix: str = "📊 全球新闻分析报告"
    ) -> bool:
        """
        发送分析报告邮件
        
        Args:
            analysis_result: 分析结果字典
            subject_prefix: 邮件主题前缀
        
        Returns:
            是否发送成功
        """
        try:
            # 构建邮件主题
            date_str = datetime.now().strftime('%Y年%m月%d日')
            subject = f"{subject_prefix} - {date_str}"
            
            # 构建邮件内容
            html_body = self._build_html_body(analysis_result)
            
            # 发送邮件
            success = self.send_email(
                subject=subject,
                html_body=html_body,
                plain_body=self._extract_plain_text(analysis_result)
            )
            
            if success:
                logger.info("分析报告邮件发送成功")
            else:
                logger.error("分析报告邮件发送失败")
            
            return success
        
        except Exception as e:
            logger.error(f"发送分析报告失败: {e}")
            return False
    
    def send_email(
        self,
        subject: str,
        html_body: str,
        plain_body: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """
        发送邮件（支持多个收件人）
        
        Args:
            subject: 邮件主题
            html_body: HTML 格式邮件内容
            plain_body: 纯文本格式邮件内容
            attachments: 附件文件路径列表
        
        Returns:
            是否发送成功
        """
        if not self.recipient_emails:
            logger.error("没有配置收件人邮箱")
            return False
        
        try:
            # 创建邮件对象
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)  # 支持多个收件人
            msg['Subject'] = subject
            
            # 添加纯文本内容
            if plain_body:
                part1 = MIMEText(plain_body, 'plain', 'utf-8')
                msg.attach(part1)
            
            # 添加 HTML 内容
            part2 = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(part2)
            
            # 添加附件
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        self._attach_file(msg, file_path)
            
            # 连接 SMTP 服务器并发送
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # 启用 TLS
                server.login(self.sender_email, self.sender_password)
                # 向所有收件人发送
                server.send_message(msg, to_addrs=self.recipient_emails)
            
            logger.info(f"邮件发送成功: {subject} -> {', '.join(self.recipient_emails)}")
            return True
        
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP 认证失败: {e}")
            logger.error("请检查邮箱地址和密码是否正确。Gmail 需要使用应用专用密码。")
            return False
        
        except smtplib.SMTPException as e:
            logger.error(f"SMTP 错误: {e}")
            return False
        
        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            return False
    
    def _build_html_body(self, analysis_result: Dict[str, Any]) -> str:
        """
        构建 HTML 格式的邮件内容
        
        Args:
            analysis_result: 分析结果
        
        Returns:
            HTML 字符串
        """
        analysis_text = analysis_result.get('analysis', '暂无分析')
        articles_count = analysis_result.get('articles_count', 0)
        history_days = analysis_result.get('history_days', 0)
        history_count = analysis_result.get('history_articles_count', 0)
        regions = ', '.join(analysis_result.get('regions_covered', []))
        analysis_time = analysis_result.get('analysis_time', '')

        # 将 Markdown 格式的分析转换为 HTML
        analysis_html = self._markdown_to_html(analysis_text)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            border-bottom: 3px solid #2196F3;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2196F3;
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }}
        .meta-item {{
            display: inline-block;
            margin-right: 20px;
            padding: 5px 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }}
        h2 {{
            color: #1976D2;
            border-left: 4px solid #2196F3;
            padding-left: 15px;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        h3 {{
            color: #1565C0;
            border-left: 4px solid #2196F3;
            padding-left: 14px;
            margin-top: 28px;
            margin-bottom: 12px;
            font-size: 17px;
        }}
        h4 {{
            color: #37474F;
            border-left: 3px solid #90CAF9;
            padding-left: 10px;
            margin-top: 18px;
            margin-bottom: 8px;
            font-size: 15px;
        }}
        ul, ol {{
            padding-left: 25px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        .analysis-content {{
            background-color: #fafafa;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
        .important {{
            background-color: #FFF3CD;
            border-left: 4px solid #FFC107;
            padding: 15px;
            margin: 20px 0;
        }}
        .success {{
            background-color: #D4EDDA;
            border-left: 4px solid #28A745;
            padding: 15px;
            margin: 20px 0;
        }}
        .warning {{
            background-color: #F8D7DA;
            border-left: 4px solid #DC3545;
            padding: 15px;
            margin: 20px 0;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        blockquote {{
            border-left: 4px solid #2196F3;
            margin: 16px 0;
            padding: 10px 16px;
            background-color: #E3F2FD;
            color: #555;
            border-radius: 0 4px 4px 0;
        }}
        blockquote p {{
            margin: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 14px;
        }}
        th {{
            background-color: #1976D2;
            color: white;
            padding: 10px 14px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 9px 14px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
        }}
        tr:nth-child(even) td {{
            background-color: #f5f8ff;
        }}
        tr:hover td {{
            background-color: #e8f0fe;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e0e0e0;
            margin: 24px 0;
        }}
        strong {{
            color: #222;
        }}
        p {{
            margin: 6px 0 10px 0;
        }}
        li p {{
            margin: 0;
        }}
        em {{
            color: #555;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 全球新闻分析报告</h1>
            <div class="meta">
                <span class="meta-item">📅 {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</span>
                <span class="meta-item">📰 今日新闻: {articles_count} 篇</span>
                {f'<span class="meta-item">📂 历史参考: {history_days} 天 / {history_count} 篇</span>' if history_days > 0 else ''}
                <span class="meta-item">🌍 覆盖地区: {regions}</span>
            </div>
        </div>
        
        <div class="analysis-content">
            {analysis_html}
        </div>
        
        {self._build_references_html(analysis_result)}
        
        <div class="footer">
            <p>本报告由 AI 自动生成，仅供参考，不构成投资建议</p>
            <p>生成时间: {analysis_time}</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """
        将 Markdown 文本转换为 HTML。

        预处理：
          - 剥除 LLM 偶尔在回复外层包裹的 ```markdown ... ``` 代码围栏
          - 去除首尾多余空白

        启用扩展（via `extra` 包 + 单独扩展）：
          extra      = tables + fenced_code + footnotes + attr_list +
                       def_list + abbr + md_in_html
          sane_lists = 更严格的列表解析，避免缩进歧义
          smarty     = 智能引号/破折号（--- → —）

        注意：故意不启用 nl2br，避免在长列表项续行处插入多余 <br>，
        行间距由 CSS 的 p/li margin 控制。

        Args:
            markdown_text: Markdown 格式字符串

        Returns:
            HTML 字符串
        """
        import re

        # 剥除整体 ```markdown ... ``` 或 ``` ... ``` 包裹
        text = markdown_text.strip()
        text = re.sub(
            r'^```(?:markdown)?\s*\n([\s\S]*?)\n```\s*$',
            r'\1',
            text,
            flags=re.MULTILINE,
        )

        extensions = [
            'extra',       # tables, fenced_code, footnotes, attr_list, def_list, abbr
            'sane_lists',  # 严格列表解析
            'smarty',      # 智能标点
        ]
        extension_configs = {
            'smarty': {
                'smart_quotes': False,   # 不转换引号（避免中文引号被替换）
                'smart_dashes': True,    # --- → —
            },
        }
        return md_lib.markdown(
            text,
            extensions=extensions,
            extension_configs=extension_configs,
            output_format='html',
        )
    
    def _build_references_html(self, analysis_result: Dict[str, Any]) -> str:
        """
        构建参考资料的 HTML 内容
        
        Args:
            analysis_result: 分析结果
        
        Returns:
            参考资料的 HTML 字符串
        """
        all_articles = analysis_result.get('all_articles', [])
        if not all_articles:
            return ""
        
        # 按区域分组
        regions = {}
        for article in all_articles:
            region = article.get('region', 'unknown')
            if region not in regions:
                regions[region] = []
            regions[region].append(article)
        
        # 区域名称映射
        region_names = {
            'americas': '美洲地区',
            'europe': '欧洲地区',
            'asia': '亚洲地区',
            'russia': '俄罗斯及独联体',
            'global': '全球综合',
            'unknown': '其他'
        }
        
        html_parts = [
            '<div class="references" style="margin-top: 40px; padding-top: 30px; border-top: 2px solid #e0e0e0;">',
            '<h2 style="color: #1976D2;">📚 参考资料</h2>',
            f'<p style="color: #666; margin-bottom: 20px;">本报告基于以下 {len(all_articles)} 篇新闻进行分析：</p>'
        ]
        
        for region in sorted(regions.keys()):
            articles = regions[region]
            region_name = region_names.get(region, region)
            
            html_parts.append(f'<h3 style="color: #424242; margin-top: 25px; margin-bottom: 15px;">🌐 {region_name}</h3>')
            html_parts.append('<ol style="padding-left: 25px;">')
            
            for article in articles:
                title = article.get('title', '无标题')
                url = article.get('url', '')
                source = article.get('source', '未知来源')
                published_at = article.get('published_at', '')
                
                html_parts.append('<li style="margin-bottom: 15px;">')
                html_parts.append(f'<strong style="color: #333;"><a href="{url}" style="color: #2196F3; text-decoration: none;">{title}</a></strong><br>')
                html_parts.append(f'<span style="color: #666; font-size: 14px;">来源: {source}')
                if published_at:
                    html_parts.append(f' | 时间: {published_at}')
                html_parts.append('</span>')
                html_parts.append('</li>')
            
            html_parts.append('</ol>')
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _extract_plain_text(self, analysis_result: Dict[str, Any]) -> str:
        """
        提取纯文本内容
        
        Args:
            analysis_result: 分析结果
        
        Returns:
            纯文本字符串
        """
        analysis_text = analysis_result.get('analysis', '暂无分析')
        articles_count = analysis_result.get('articles_count', 0)
        regions = ', '.join(analysis_result.get('regions_covered', []))
        all_articles = analysis_result.get('all_articles', [])
        
        # 构建参考资料文本
        references_text = ""
        if all_articles:
            references_text = f"\n\n{'=' * 60}\n\n参考资料\n\n本报告基于以下 {len(all_articles)} 篇新闻进行分析：\n\n"
            
            # 按区域分组
            regions_dict = {}
            for article in all_articles:
                region = article.get('region', 'unknown')
                if region not in regions_dict:
                    regions_dict[region] = []
                regions_dict[region].append(article)
            
            region_names = {
                'americas': '美洲地区',
                'europe': '欧洲地区',
                'asia': '亚洲地区',
                'russia': '俄罗斯及独联体',
                'global': '全球综合',
                'unknown': '其他'
            }
            
            for region in sorted(regions_dict.keys()):
                articles_list = regions_dict[region]
                region_name = region_names.get(region, region)
                references_text += f"\n{region_name}\n\n"
                
                for i, article in enumerate(articles_list, 1):
                    title = article.get('title', '无标题')
                    url = article.get('url', '')
                    source = article.get('source', '未知来源')
                    references_text += f"{i}. {title}\n   来源: {source}\n   链接: {url}\n\n"
        
        return f"""
全球新闻分析报告
{datetime.now().strftime('%Y年%m月%d日 %H:%M')}

分析文章数: {articles_count}
覆盖地区: {regions}

{'-' * 60}

{analysis_text}

{references_text}
{'-' * 60}

本报告由 AI 自动生成，仅供参考，不构成投资建议
"""
    
    def _attach_file(self, msg: MIMEMultipart, file_path: str):
        """
        添加附件到邮件
        
        Args:
            msg: 邮件对象
            file_path: 文件路径
        """
        try:
            filename = os.path.basename(file_path)
            with open(file_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {filename}')
            msg.attach(part)
            
            logger.info(f"添加附件: {filename}")
        
        except Exception as e:
            logger.error(f"添加附件失败 ({file_path}): {e}")

