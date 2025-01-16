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
stock_code_list = ['600733.SH','000012.SZ','002171.SZ','002929.SZ','002414.SZ']
prices = pd.DataFrame()  # 创建一个空的DataFrame，用于存放股票价格数据
for stock_code in stock_code_list:
    if 'trade_date' not in prices.columns:
        prices['trade_date'] = ts.pro_bar(ts_code=stock_code, api=pro, adj='qfq', start_date='20230215')['trade_date']
    prices[stock_code] = ts.pro_bar(ts_code=stock_code, api=pro, adj='qfq', start_date='20230215')['close']

# 数据清洗及处理
prices.dropna(inplace=True)  # 数据清洗
prices.set_index('trade_date', inplace=True)  # 设置索引
prices.sort_index(ascending=True, inplace=True)  # 重新排序

# 基础数组创建
asset_nums = prices.shape[1]  # 资产个数
returns = np.log(prices/prices.shift(1))  # 创建偏移后的数组

# 定义函数
import math
def port_rets(weights):  # 组合年化收益率
    return np.sum(returns.mean() * weights) * 252
def port_vols(weights):  # 组合年化标准差（波动率）
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

# 最大化夏普比率
import scipy.optimize as sco
def min_func_sharpe(weights):
    return -port_rets(weights)/port_vols(weights)
cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 定义约束：权重和为1
bnds = tuple((0, 1) for x in range(asset_nums))  # 取值区间
weights0 = np.array(asset_nums * [1./asset_nums, ])  # 初始权重
opts = sco.minimize(min_func_sharpe, weights0, bounds=bnds, constraints=cons, method='SLSQP')  # 条件函数

# 最小方差组合
optv = sco.minimize(port_vols, weights0, bounds=bnds, constraints=cons, method='SLSQP')

# 最小方差前沿与有效前沿
cons = ({'type': 'eq', 'fun': lambda x: port_rets(x) - tret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 约束条件：组合收益等于目标收益，权重和为1
bnds = tuple((0, 1) for x in range(asset_nums))  # 取值区间
weights00 = np.array(asset_nums * [1./asset_nums, ])  # 初始值
trets = np.linspace(-0.5, port_rets(opts['x']), 50)  # 生成序列范围
tvols = []
weights = []
for tret in trets:
    res = sco.minimize(port_vols, weights00, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
    weights.append(res['x'])
tvols = np.array(tvols)  # 转换为数组


ind = np.argmin(tvols)
evols = tvols[ind:]  # 最小值后面的值
erets = trets[ind:]  # 最小值后面的值


# 画图
plt.figure(figsize=(10, 8))
plt.plot(evols, erets, 'b', lw=4.0)
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()