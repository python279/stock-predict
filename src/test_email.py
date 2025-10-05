#!/usr/bin/env python3
"""
邮件发送测试工具
"""

import sys
from config_loader import ConfigLoader
from email_sender import EmailSender
from datetime import datetime

def main():
    print("="*60)
    print("邮件发送测试工具")
    print("="*60)
    
    # 加载配置
    try:
        config = ConfigLoader("config.yaml").config
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 显示邮件配置
    email_config = config.get('email', {})
    print(f"\n邮件配置:")
    print(f"  SMTP服务器: {email_config.get('smtp_server')}")
    print(f"  SMTP端口: {email_config.get('smtp_port')}")
    print(f"  发件人: {email_config.get('sender_email')}")
    print(f"  收件人: {email_config.get('recipient_email')}")
    
    # 初始化邮件发送器
    try:
        sender = EmailSender(config)
        print("✅ 邮件发送器初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    # 发送测试邮件
    print("\n正在发送测试邮件...")
    
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2196F3;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 10px;
        }}
        .success {{
            background-color: #D4EDDA;
            border-left: 4px solid #28A745;
            padding: 15px;
            margin: 20px 0;
        }}
        .info {{
            color: #666;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>✅ 邮件发送测试</h1>
        
        <div class="success">
            <h2>恭喜！</h2>
            <p>如果你收到这封邮件，说明邮件配置成功！</p>
        </div>
        
        <p><strong>测试时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h3>测试项目：</h3>
        <ul>
            <li>✅ SMTP 连接成功</li>
            <li>✅ 身份认证通过</li>
            <li>✅ 邮件发送成功</li>
            <li>✅ HTML 格式渲染正常</li>
        </ul>
        
        <div class="info">
            <p>现在你可以正常运行全球新闻分析系统了！</p>
            <p>运行命令: <code>.venv/bin/python3 main.py</code></p>
        </div>
    </div>
</body>
</html>
"""
    
    plain_body = f"""
邮件发送测试

如果你收到这封邮件，说明邮件配置成功！

测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

测试项目:
- SMTP 连接成功
- 身份认证通过
- 邮件发送成功

现在你可以正常运行全球新闻分析系统了！
"""
    
    try:
        success = sender.send_email(
            subject="🎉 全球新闻分析系统 - 邮件测试",
            html_body=html_body,
            plain_body=plain_body
        )
        
        if success:
            print("\n" + "="*60)
            print("✅ 测试邮件发送成功！")
            print("="*60)
            print("\n请检查你的邮箱收件箱（如果没有收到，检查垃圾邮件文件夹）")
            print(f"收件箱: {email_config.get('recipient_email')}")
            return True
        else:
            print("\n" + "="*60)
            print("❌ 测试邮件发送失败")
            print("="*60)
            print("\n可能的原因：")
            print("1. SMTP 服务器连接超时")
            print("2. 用户名或密码错误")
            print("3. 未开启 SMTP 服务")
            print("4. 防火墙拦截")
            print("\n请查看日志获取详细信息: ../logs/news_analyzer.log")
            print("\n建议切换到QQ邮箱（参考 README.md）")
            return False
    
    except Exception as e:
        print(f"\n❌ 发送过程出错: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(130)

