"""
摩尔线程(688795) 定量分析脚本
=================================
用途: 对腾讯财经K线数据进行统计分析
输入: 从腾讯API获取的JSON数据 (web.ifzq.gtimg.cn)
用法: 粘贴数据到 data 变量后运行

API: curl -s "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh{CODE},day,,,120,qfq"
"""

import json
from datetime import datetime

# ============================================================
# 1. 粘贴数据 — 从腾讯API返回的完整JSON
# ============================================================
RAW_JSON = '{"data":{"sh688795":{"day":[]}}}'  # 替换为实际数据

data = json.loads(RAW_JSON)
code = list(data['data'].keys())[0]
days = data['data'][code]['day']

# ============================================================
# 2. 解析
# ============================================================
parsed = []
for d in days:
    parsed.append({
        'date': d[0],
        'open': float(d[1]),
        'close': float(d[2]),
        'high': float(d[3]),
        'low': float(d[4]),
        'vol': float(d[5])
    })

print(f"交易日数: {len(parsed)}")
print(f"日期范围: {parsed[0]['date']} ~ {parsed[-1]['date']}")
print()

# ============================================================
# 3. 收益率统计
# ============================================================
returns = []
for i in range(1, len(parsed)):
    ret = (parsed[i]['close'] - parsed[i-1]['close']) / parsed[i-1]['close'] * 100
    returns.append({'date': parsed[i]['date'], 'ret': ret})

rets = [r['ret'] for r in returns]
print("=== 收益率统计 ===")
print(f"平均日收益率: {sum(rets)/len(rets):+.2f}%")
print(f"最大日涨幅: {max(rets):+.2f}%")
print(f"最大日跌幅: {min(rets):+.2f}%")
print(f"上涨天数: {sum(1 for r in rets if r>0)}/{len(rets)} ({sum(1 for r in rets if r>0)/len(rets)*100:.1f}%)")
print()

# ============================================================
# 4. 星期几效应
# ============================================================
wd = {0:[], 1:[], 2:[], 3:[], 4:[]}
wd_names = {0:'周一', 1:'周二', 2:'周三', 3:'周四', 4:'周五'}
for i in range(1, len(parsed)):
    dt = datetime.strptime(parsed[i]['date'], '%Y-%m-%d')
    ret = (parsed[i]['close'] - parsed[i-1]['close']) / parsed[i-1]['close'] * 100
    wd[dt.weekday()].append(ret)

print("=== 星期几效应 ===")
for d in range(5):
    if wd[d]:
        avg = sum(wd[d])/len(wd[d])
        win = sum(1 for r in wd[d] if r>0)/len(wd[d])*100
        print(f"{wd_names[d]}: 平均{avg:+.2f}%, 胜率{win:.1f}% (n={len(wd[d])})")
print()

# ============================================================
# 5. 均线计算
# ============================================================
closes = [p['close'] for p in parsed]
print("=== 均线系统 ===")
for n in [5, 10, 20, 30]:
    if len(closes) >= n:
        ma = sum(closes[-n:]) / n
        direction = '⬆ 高于' if closes[-1] > ma else '⬇ 低于'
        print(f"MA{n}: {ma:.2f} ({direction})")
print()

# ============================================================
# 6. 暴跌反弹统计 (跌幅≥3%)
# ============================================================
drops = []
for i in range(len(returns)-1):
    if returns[i]['ret'] <= -3:
        drops.append(returns[i+1]['ret'])
if drops:
    bounce = sum(1 for d in drops if d>0)
    print("=== 暴跌反弹统计 (跌幅≥3%) ===")
    print(f"次数: {len(drops)}, 次日反弹概率: {bounce/len(drops)*100:.1f}%, 平均次日收益: {sum(drops)/len(drops):+.2f}%")
print()

# ============================================================
# 7. 暴涨持续统计 (涨幅≥3%)
# ============================================================
rises = []
for i in range(len(returns)-1):
    if returns[i]['ret'] >= 3:
        rises.append(returns[i+1]['ret'])
if rises:
    cont = sum(1 for r in rises if r>0)
    print("=== 暴涨持续统计 (涨幅≥3%) ===")
    print(f"次数: {len(rises)}, 次日持续概率: {cont/len(rises)*100:.1f}%, 平均次日收益: {sum(rises)/len(rises):+.2f}%")
print()

# ============================================================
# 8. 支撑/压力位 (近60日)
# ============================================================
recent = parsed[-60:]
sorted_lows = sorted([p['low'] for p in recent])
sorted_highs = sorted([p['high'] for p in recent])
print("=== 支撑/压力位 (近60日) ===")
print(f"强支撑 (20%分位): {sorted_lows[int(len(sorted_lows)*0.2)]:.2f}")
print(f"中支撑 (40%分位): {sorted_lows[int(len(sorted_lows)*0.4)]:.2f}")
print(f"中压力 (60%分位): {sorted_highs[int(len(sorted_highs)*0.6)]:.2f}")
print(f"强压力 (80%分位): {sorted_highs[int(len(sorted_highs)*0.8)]:.2f}")
print()

# ============================================================
# 9. 近期走势
# ============================================================
print("=== 近期走势 (近10日) ===")
for p in parsed[-10:]:
    print(f"{p['date']} O:{p['open']:.2f} H:{p['high']:.2f} L:{p['low']:.2f} C:{p['close']:.2f}")
