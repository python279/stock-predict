"""
配置测试脚本
用于验证配置是否正确，各个模块是否能正常初始化
"""

import sys
import logging
from typing import Dict, Any

# 设置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_config_loading():
    """测试配置加载"""
    print("\n" + "="*60)
    print("测试 1: 配置加载")
    print("="*60)
    try:
        from config_loader import ConfigLoader
        config_loader = ConfigLoader("config.yaml")
        print("✅ 配置文件加载成功")
        
        # 验证关键配置
        print("\n关键配置检查:")
        checks = [
            ('email.sender_email', '发件邮箱'),
            ('email.sender_password', '邮箱密码'),
            ('email.recipient_email', '收件邮箱'),
            ('llm.api_key', 'LLM API Key'),
            ('llm.model', 'LLM 模型'),
        ]
        
        all_valid = True
        for key, desc in checks:
            value = config_loader.get(key)
            if not value or str(value).startswith('your-'):
                print(f"  ⚠️  {desc} ({key}): 未配置或使用默认值")
                all_valid = False
            else:
                masked_value = str(value)[:10] + "..." if len(str(value)) > 10 else str(value)
                print(f"  ✅ {desc} ({key}): {masked_value}")
        
        return all_valid
    
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_news_fetcher():
    """测试新闻抓取器初始化"""
    print("\n" + "="*60)
    print("测试 2: 新闻抓取器初始化")
    print("="*60)
    try:
        from config_loader import ConfigLoader
        from news_fetcher import NewsFetcher
        
        config = ConfigLoader("config.yaml").config
        fetcher = NewsFetcher(config)
        print("✅ 新闻抓取器初始化成功")
        
        # 显示配置信息
        newsapi_enabled = config.get('news_api', {}).get('enabled', False)
        rss_enabled = config.get('rss_feeds', {}).get('enabled', False)
        rss_count = len(config.get('rss_feeds', {}).get('sources', []))
        
        print(f"\n抓取源配置:")
        print(f"  News API: {'启用' if newsapi_enabled else '禁用'}")
        print(f"  RSS Feed: {'启用' if rss_enabled else '禁用'} ({rss_count} 个源)")
        
        return True
    
    except Exception as e:
        print(f"❌ 新闻抓取器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_analyzer():
    """测试 LLM 分析器初始化"""
    print("\n" + "="*60)
    print("测试 3: LLM 分析器初始化")
    print("="*60)
    try:
        from config_loader import ConfigLoader
        from llm_analyzer import LLMAnalyzer
        
        config = ConfigLoader("config.yaml").config
        analyzer = LLMAnalyzer(config)
        print("✅ LLM 分析器初始化成功")
        
        # 显示配置信息
        provider = config.get('llm', {}).get('provider', 'unknown')
        model = config.get('llm', {}).get('model', 'unknown')
        base_url = config.get('llm', {}).get('base_url', '')
        
        print(f"\nLLM 配置:")
        print(f"  提供商: {provider}")
        print(f"  模型: {model}")
        if base_url:
            print(f"  Base URL: {base_url}")
        
        return True
    
    except Exception as e:
        print(f"❌ LLM 分析器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_email_sender():
    """测试邮件发送器初始化"""
    print("\n" + "="*60)
    print("测试 4: 邮件发送器初始化")
    print("="*60)
    try:
        from config_loader import ConfigLoader
        from email_sender import EmailSender
        
        config = ConfigLoader("config.yaml").config
        email_sender = EmailSender(config)
        print("✅ 邮件发送器初始化成功")
        
        # 显示配置信息
        smtp_server = config.get('email', {}).get('smtp_server', '')
        smtp_port = config.get('email', {}).get('smtp_port', 0)
        sender = config.get('email', {}).get('sender_email', '')
        recipient = config.get('email', {}).get('recipient_email', '')
        
        print(f"\n邮件配置:")
        print(f"  SMTP 服务器: {smtp_server}:{smtp_port}")
        print(f"  发件人: {sender}")
        print(f"  收件人: {recipient}")
        
        return True
    
    except Exception as e:
        print(f"❌ 邮件发送器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_fetch():
    """快速测试抓取一条新闻"""
    print("\n" + "="*60)
    print("测试 5: 快速抓取测试")
    print("="*60)
    
    response = input("是否进行实际抓取测试？这会消耗 API 配额 (y/N): ")
    if response.lower() != 'y':
        print("⏭️  跳过抓取测试")
        return True
    
    try:
        from config_loader import ConfigLoader
        from news_fetcher import NewsFetcher
        
        config = ConfigLoader("config.yaml").config
        fetcher = NewsFetcher(config)
        
        print("\n正在抓取新闻（可能需要一些时间）...")
        articles = fetcher.fetch_all_news()
        
        if articles:
            print(f"✅ 成功抓取 {len(articles)} 篇文章")
            print("\n前3篇文章示例:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n{i}. {article.title}")
                print(f"   来源: {article.source}")
                print(f"   地区: {article.region}")
                print(f"   时间: {article.published_at}")
            return True
        else:
            print("⚠️  未抓取到文章")
            return False
    
    except Exception as e:
        print(f"❌ 抓取测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "🔍 " * 20)
    print("全球新闻分析系统 - 配置测试工具")
    print("🔍 " * 20)
    
    # 运行所有测试
    results = []
    
    results.append(("配置加载", test_config_loading()))
    results.append(("新闻抓取器", test_news_fetcher()))
    results.append(("LLM 分析器", test_llm_analyzer()))
    results.append(("邮件发送器", test_email_sender()))
    results.append(("抓取测试", test_quick_fetch()))
    
    # 显示总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n总计: {success_count}/{total_count} 项测试通过")
    
    if success_count == total_count:
        print("\n🎉 所有测试通过！系统已准备就绪。")
        print("\n下一步:")
        print("  1. 运行: cd ../src && python main.py  (立即执行)")
        print("  2. 运行: cd ../src && python scheduler.py  (启动调度器)")
        print("  3. 运行: cd .. && ./run.sh  (使用启动脚本)")
    else:
        print("\n⚠️  部分测试失败，请检查配置和依赖安装。")
    
    return success_count == total_count


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

