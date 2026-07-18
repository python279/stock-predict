# 东方财富 Push API 字段参考

> 端点: `https://push2.eastmoney.com/api/qt/stock/get?secid=1.{CODE}&fields={FIELDS}`
> 端点: `https://push2.eastmoney.com/api/qt/stock/gett?secid=1.{CODE}&fields={FIELDS}` (精简版)
> 来源: 东方财富移动端/推送接口，无需认证，无CAPTCHA

## 字段映射表

| 字段 | 含义 | 单位 | 需÷100? |
|------|------|------|---------|
| f43 | 最新价 (current price) | 分 | ✅ |
| f44 | 今开 (open) | 分 | ✅ |
| f45 | 最高 (high) | 分 | ✅ |
| f46 | 最低 (low) | 分 | ✅ |
| f47 | 成交量 (volume) | 手 (1手=100股) | ❌ |
| f48 | 成交额 (turnover) | 元 | ❌ |
| f49 | 换手率 (turnover rate) | 百分比×100 | ✅ |
| f50 | 振幅 (amplitude) | 百分比×100 | ✅ |
| f51 | 涨停价 (limit up) | 分 | ✅ |
| f52 | 跌停价 (limit down) | 分 | ✅ |
| f55 | 涨跌幅 (change %) | 小数 (如0.05=5%) | ❌ (直接使用) |
| f57 | 股票代码 | 字符串 | ❌ |
| f58 | 股票名称 | 字符串 | ❌ |
| f60 | 昨收 (prev close) | 分 | ✅ |
| f116 | 总市值 (total mkt cap) | 元 | ❌ |
| f117 | 流通市值 (float mkt cap) | 元 | ❌ |
| f162 | 市盈率(动) (PE TTM) | 倍数×100 | ✅ |
| f167 | 上涨家数 | 整数 | ❌ |
| f168 | 下跌家数 | 整数 | ❌ |
| f169 | 未知 (可能为涨停家数) | 整数 | ❌ |
| f170 | 平盘家数 | 整数 | ❌ |
| f171 | 总家数 | 整数 | ❌ |
| f292 | 交易状态 | 1=盘前,2=交易中,3=收盘 | ❌ |

## 常用字段组合

### 基本行情 (推荐)
```
f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162
```

### 完整行情 (含家数统计)
```
f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162,f167,f168,f169,f170,f171
```

### 精简版 (仅价格+成交量)
```
f43,f47,f48,f55,f57,f58,f60
```

## 使用示例

```bash
# 获取摩尔线程(688795)完整行情
curl -s "https://push2.eastmoney.com/api/qt/stock/get?secid=1.688795&fields=f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162,f167,f168,f169,f170,f171" -H "User-Agent: Mozilla/5.0"

# 解析示例 (Python)
import json
data = json.loads(output)
d = data['data']
price = d['f43'] / 100
change_pct = d['f55'] * 100  # 如 5.23 表示 +5.23%
turnover_rate = d['f49'] / 100  # 如 11.80 表示 11.80%
pe = d['f162'] / 100
```

## 注意

- 科创板/创业板股票使用 `secid=1.{CODE}` (深交所统一编码)
- 上证主板使用 `secid=1.{CODE}` 同样适用
- 港股使用 `secid=0.{CODE}` (五位代码)
- 所有价格字段返回的是 **分** 单位，必须 ÷100 得到元
- 涨跌幅 f55 是小数形式 (0.05=5%)，直接使用
- 换手率 f49 是百分比×100 (如 1180 = 11.80%)，需 ÷100
