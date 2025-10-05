"""
RSS源测试工具
测试所有配置的RSS源是否可用
"""

import feedparser
import sys
import time
from config_loader import ConfigLoader

def test_rss_source(name, url, timeout=10):
    """测试单个RSS源"""
    print(f"\n测试: {name}")
    print(f"URL: {url}")
    
    try:
        # 设置超时
        import socket
        socket.setdefaulttimeout(timeout)
        
        # 解析RSS
        feed = feedparser.parse(url)
        
        # 检查是否成功
        if feed.bozo:
            print(f"⚠️  解析警告: {feed.bozo_exception}")
        
        # 检查条目数
        entry_count = len(feed.entries)
        if entry_count > 0:
            print(f"✅ 成功 - 获取到 {entry_count} 条新闻")
            # 显示第一条新闻标题
            if feed.entries:
                first_title = feed.entries[0].get('title', 'No title')
                print(f"   示例: {first_title[:80]}...")
            return True
        else:
            print(f"❌ 失败 - 没有获取到新闻条目")
            return False
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("RSS 新闻源测试工具")
    print("="*60)
    
    # 加载配置
    try:
        config = ConfigLoader("config.yaml").config
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        sys.exit(1)
    
    # 获取RSS源
    rss_sources = config.get('rss_feeds', {}).get('sources', [])
    
    if not rss_sources:
        print("❌ 没有配置RSS源")
        sys.exit(1)
    
    print(f"\n共有 {len(rss_sources)} 个RSS源需要测试\n")
    
    # 测试所有源
    results = []
    for i, source in enumerate(rss_sources, 1):
        name = source.get('name', 'Unknown')
        url = source.get('url', '')
        region = source.get('region', 'unknown')
        
        if not url:
            print(f"\n{i}. {name} - ❌ URL为空")
            results.append((name, False, region))
            continue
        
        print(f"\n{'='*60}")
        print(f"{i}/{len(rss_sources)}")
        success = test_rss_source(name, url)
        results.append((name, success, region))
        
        # 避免请求过快
        time.sleep(1)
    
    # 显示总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    print(f"\n总计: {success_count}/{total_count} 个源可用\n")
    
    # 按地区分类
    regions = {}
    for name, success, region in results:
        if region not in regions:
            regions[region] = []
        regions[region].append((name, success))
    
    for region, sources in regions.items():
        print(f"\n{region.upper()}:")
        for name, success in sources:
            status = "✅" if success else "❌"
            print(f"  {status} {name}")
    
    # 列出失败的源
    failed = [(name, region) for name, success, region in results if not success]
    if failed:
        print("\n需要修复的源:")
        for name, region in failed:
            print(f"  - {name} ({region})")
    
    print("\n" + "="*60)
    
    return success_count == total_count


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(130)

