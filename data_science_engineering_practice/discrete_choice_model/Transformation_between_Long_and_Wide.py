# 数据格式的清洗以及数据预处理,以及长宽表的相互转换函数

import pandas as pd
import numpy as np

import statsmodels
import pylogit as pl  # collections版本问题要进行修正 库中的相应的import要改为 from collections.abc import Iterable
from collections import OrderedDict  # 用于记录模型的信息


# 长格式转换为宽格式
def long_format_to_wide(data_path):
    long_df = pd.read_csv(data_path, sep=',', header=0)
    print(long_df.head())

    # 转换流程
    # 决策者属性的列
    individual_specific_variables = ['HINC', 'PSIZE']  # 家庭收入以及人数
    # 指定备选项属性的列表
    alternative_specific_variables = ['TTME', 'INVC', 'INVT', 'GC']
    # 指定备选项属性的特殊说明列表
    subset_specific_variables = {}
    # “观测者ID”，标识每次选择。
    observation_id_column = "OBS_ID"
    # “备选项ID”，标识备选方案。
    alternative_id_column = "ALT_ID"
    # “选择结果”，标识选择结果
    choice_column = "MODE"

    # 可选变量，选项的标签化
    alternative_name_dict = {
        0: "AIR",
        1: "TRAIN",
        2: "BUS",
        3: "CAR"
    }
    # 转化为宽表
    wide_df = pl.convert_long_to_wide(long_df,
                                      individual_specific_variables,
                                      alternative_specific_variables,
                                      subset_specific_variables,
                                      observation_id_column,
                                      alternative_id_column,
                                      choice_column,
                                      alternative_name_dict)

    return wide_df


# 宽格式转换为长格式
def wide_format_to_long(data_path):
    wide_df = pd.read_csv(data_path, sep=',', header=0)
    print(wide_df.head())

    # 创建决策者变量的列表
    ind_variables = ["HINC", "PSIZE"]
    # 指定每个备选项的属性所对应的字段，
    # 0/1/2/3代表备选项，后面的字段名为其属性在“宽格式数据“中的列名
    alt_varying_variables = {"TTME": dict([(0, 'TTME_AIR'),
                                           (1, 'TTME_TRAIN'),
                                           (2, 'TTME_BUS'),
                                           (3, 'TTME_CAR')]),
                             "INVC": dict([(0, 'INVC_AIR'),
                                           (1, 'INVC_TRAIN'),
                                           (2, 'INVC_BUS'),
                                           (3, 'INVC_CAR')]),
                             "INVT": dict([(0, 'INVT_AIR'),
                                           (1, 'INVT_TRAIN'),
                                           (2, 'INVT_BUS'),
                                           (3, 'INVT_CAR')]),
                             "GC": dict([(0, 'GC_AIR'),
                                         (1, 'GC_TRAIN'),
                                         (2, 'GC_BUS'),
                                         (3, 'GC_CAR')])}

    # 指定可用性变量注意字典的键是可选的id。这些值是表示数据集中给定模式可用性的列。
    # 前面的宽格式数据中省略了这3列，实际在做数据转化时需要标示每个选项的可用性。
    availability_variables = {0: 'availability_AIR',
                              1: 'availability_TRAIN',
                              2: 'availability_BUS',
                              3: 'availability_CAR'}
    # “备选项ID”，标识与每一行相关联的备选方案。
    custom_alt_id = "ALT_ID"
    # “观测ID”，标识每次选择。注意+1确保id从1开始。
    obs_id_column = "OBS_ID"
    wide_df[obs_id_column] = np.arange(wide_df.shape[0], dtype=int) + 1
    # “选择结果”，标识选择结果
    choice_column = 'MODE'
    # 执行长格式转换
    long_df = pl.convert_wide_to_long(wide_df,
                                      ind_variables,
                                      alt_varying_variables,
                                      availability_variables,
                                      obs_id_column,
                                      choice_column,
                                      new_alt_id_name=custom_alt_id)

    return long_df


if __name__ == '__main__':
    # 使用的数据是量化经济学建模中的旅行选择问题数据
    # 读取数据
    data_path = u'data/long_data.csv'
    # use
    df = long_format_to_wide(data_path)

    print(df.head())
