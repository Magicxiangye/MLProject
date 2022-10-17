# 数据格式的清洗以及数据预处理,以及长宽表的相互转换函数

import pandas as pd

import statsmodels
import pylogit as pl  # collections版本问题要进行修正 库中的相应的import要改为 from collections.abc import Iterable
from collections import OrderedDict  # 用于记录模型的信息


# 长格式转换为宽格式
def long_format_to_wide(data_path):
    row_data = pd.read_csv(data_path, sep=',', header=0)
    print(row_data.head())

    # 转换流程
    # 决策者属性的列
    individual_specific_variables = ['HINC', 'PSIZE']  # 家庭收入以及人数



if __name__ == '__main__':
    # 使用的数据是量化经济学建模中的旅行选择问题数据
    # 读取数据
    data_path = u'data/long_data.csv'
    # use
    long_format_to_wide(data_path)
