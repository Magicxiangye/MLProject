# pugc 达人聚类
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

# all columns
pd.set_option('display.max_columns', None)


# 最大最小的归一化
def col2Norm(data):
    return data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


# 转码数据
# 数据的读取预处理
def load_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    # 去空值
    # 剩余518个作者
    df = df.dropna().reset_index()
    return df

# EDA
def preprocessEda(feature_df):
    # 先做一个特征的探索，进行一个特征的降维处理
    # 使用方差解释率，来选择降维的维度
    pca = PCA()
    pca.fit(feature_df)

    # 设置保留整体数据集95%方差所需要的最小维度数量
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    print('ratio')
    # print(pca.explained_variance_ratio_)

    # 对应原始变量的系数
    # 每列累计权重最大的就是比较关键的变量
    # 主成分对应原始变量的系数
    pca_as = pca.components_

    # 最小纬度
    d = np.argmax(cumsum >= 0.95) + 1
    print(d)

    return pca_as


if __name__ == '__main__':
    file_path = 'data/classif_cloumns.csv'

    df = load_data(file_path)

    print(df.shape[0])

    feature_df = df[['5月被开练课程数量', '开练UV', '开练PV', '首次开练UV', '首次开练PV', '5月被完练课程数量', '完练UV', '完练PV',
                     '首次完练UV', '首次完练PV', '首次完练&复练UV', '创作者次月完练-完练留存UV', '详情页UV', '详情页PV', '创作者月完练-完练留存UV',
                     '创作者月完练-7日内App留存UV', '次月复练率-UV', '完练率-UV', '当月复练率', '完练-7日内App留存率']]

    # 数据量级上的标准化
    # 最大最小标准化
    feature_df = col2Norm(feature_df)
    # 降维的维度为6
    pca_as = preprocessEda(feature_df)

    pca_as = pd.DataFrame(pca_as)
    print(pca_as)

    sum_count = pca_as.iloc[:2, :].apply(lambda x: x.sum())

    print(pca_as.iloc[:2, :])

    print(sum_count)




