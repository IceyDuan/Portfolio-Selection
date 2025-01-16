import numpy as np
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

# 前置处理
pd.set_option('display.max_rows',10000)
pd.set_option('display.max_columns',10000)
np.set_printoptions(threshold=np.inf)  # 取消print结果显示限制
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题

# 数据库token
My_token = '3b903a2c73297d69aaf9caf8adc356249b787a56182cdba4e65a48a6'
pro = ts.pro_api(My_token)

# 数据导入
stock_code_list = ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ', '000008.SZ', '000009.SZ', '000010.SZ', '000011.SZ', '000012.SZ', '000014.SZ', '000016.SZ', '000017.SZ', '000019.SZ', '000020.SZ', '000021.SZ', '000023.SZ', '000025.SZ', '000026.SZ', '000027.SZ', '000028.SZ', '000029.SZ', '000032.SZ', '000045.SZ', '000049.SZ', '000050.SZ', '000059.SZ', '000096.SZ', '600000.SH', '600004.SH', '600006.SH', '600007.SH', '600008.SH', '600009.SH', '600010.SH', '600011.SH', '600012.SH', '600015.SH', '600016.SH', '600017.SH', '600018.SH', '600019.SH', '600025.SH', '600030.SH', '600036.SH', '600038.SH', '600050.SH', '600059.SH', '600072.SH', '600088.SH']
prices = pd.DataFrame()  # 存放股票价格数据的DataFrame
for stock_code in stock_code_list:
    if 'trade_date' not in prices.columns:
        prices['trade_date'] = ts.pro_bar(ts_code=stock_code, api=pro, adj='qfq', start_date='20200109',end_date='20240228')['trade_date']
    prices[stock_code] = ts.pro_bar(ts_code=stock_code, api=pro, adj='qfq', start_date='20200109',end_date='20240228')['close']

# 数据清洗及处理
prices.dropna(inplace=True)  # 清洗
prices.set_index('trade_date', inplace=True)  # 索引
prices.sort_index(ascending=True, inplace=True)  # 排序

# 基础数组创建
asset_nums = prices.shape[1]
returns = np.log(prices/prices.shift(1))

# 定义函数
def port_rets(weights):
    return np.sum(returns.mean() * weights) * 252  # 按252个交易日处理为年化
def port_vols(weights):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # 按252个交易日处理为年化

# 最大化夏普比率
import scipy.optimize as sco
def min_func_sharpe(weights):
    return -port_rets(weights)/port_vols(weights)
cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 定义约束：权重和为1
bnds = tuple((0, 1) for x in range(asset_nums))  # 取值区间
weights0 = np.array(asset_nums * [1./asset_nums, ])  # 初始权重
opts = sco.minimize(min_func_sharpe, weights0, bounds=bnds, constraints=cons, method='SLSQP')

# 最小方差组合
optv = sco.minimize(port_vols, weights0, bounds=bnds, constraints=cons, method='SLSQP')

# 最小方差前沿与有效前沿
cons = ({'type': 'eq', 'fun': lambda x: port_rets(x) - tret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 约束条件：收益等于目标收益，权重和为1
bnds = tuple((0, 1) for x in range(asset_nums))  # 取值区间
weights00 = np.array(asset_nums * [1./asset_nums, ])  # 初始权重
trets = np.linspace(0, 0.45, 50)
tvols = []
weights = []
for tret in trets:
    res = sco.minimize(port_vols, weights00, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
    weights.append(res['x'])
tvols = np.array(tvols)
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]

# 合并结果并制表输出
returns.cov().to_excel('Covariance Table.xlsx',index=True)
np.vstack(returns.std(),returns.mean()).to_excel('Standard Deviation Table.xlsx',index=True)
prices.to_excel('Daily Closing Price Table.xlsx',index=True)
weights = np.array(weights)
weights1 = pd.DataFrame(weights)
weights1.to_excel('All Weights.xlsx', index=True)
returns.to_excel('Return.xlsx',index=True)

# 画图
plt.figure(figsize=(10, 8))
plt.plot(evols, erets, 'b', lw=4.0)
plt.plot(port_vols(opts['x']), port_rets(opts['x']), 'y*', markersize=12.0)
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()