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
        self.recipient_email = self.email_config.get('recipient_email', '')
    
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
        发送邮件
        
        Args:
            subject: 邮件主题
            html_body: HTML 格式邮件内容
            plain_body: 纯文本格式邮件内容
            attachments: 附件文件路径列表
        
        Returns:
            是否发送成功
        """
        try:
            # 创建邮件对象
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
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
                server.send_message(msg)
            
            logger.info(f"邮件发送成功: {subject}")
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
            color: #424242;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        h4 {{
            color: #616161;
            margin-top: 15px;
            margin-bottom: 8px;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 全球新闻分析报告</h1>
            <div class="meta">
                <span class="meta-item">📅 {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</span>
                <span class="meta-item">📰 分析文章: {articles_count} 篇</span>
                <span class="meta-item">🌍 覆盖地区: {regions}</span>
            </div>
        </div>
        
        <div class="analysis-content">
            {analysis_html}
        </div>
        
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
        简单的 Markdown 到 HTML 转换
        
        Args:
            markdown_text: Markdown 文本
        
        Returns:
            HTML 字符串
        """
        html_lines = []
        lines = markdown_text.split('\n')
        in_list = False
        
        for line in lines:
            line = line.rstrip()
            
            # 标题
            if line.startswith('####'):
                html_lines.append(f'<h4>{line[4:].strip()}</h4>')
            elif line.startswith('###'):
                html_lines.append(f'<h3>{line[3:].strip()}</h3>')
            elif line.startswith('##'):
                html_lines.append(f'<h2>{line[2:].strip()}</h2>')
            elif line.startswith('#'):
                html_lines.append(f'<h1>{line[1:].strip()}</h1>')
            
            # 列表
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(f'<li>{line.strip()[2:]}</li>')
            
            elif line.strip().startswith(tuple([f'{i}.' for i in range(1, 10)])):
                if not in_list:
                    html_lines.append('<ol>')
                    in_list = True
                content = line.strip().split('.', 1)[1].strip()
                html_lines.append(f'<li>{content}</li>')
            
            # 普通段落
            elif line.strip():
                if in_list:
                    html_lines.append('</ul>' if html_lines[-1].startswith('<li>') else '</ol>')
                    in_list = False
                html_lines.append(f'<p>{line}</p>')
            
            # 空行
            else:
                if in_list:
                    html_lines.append('</ul>' if '<ul>' in '\n'.join(html_lines[-10:]) else '</ol>')
                    in_list = False
                html_lines.append('<br>')
        
        if in_list:
            html_lines.append('</ul>')
        
        return '\n'.join(html_lines)
    
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
        
        return f"""
全球新闻分析报告
{datetime.now().strftime('%Y年%m月%d日 %H:%M')}

分析文章数: {articles_count}
覆盖地区: {regions}

{'-' * 60}

{analysis_text}

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

