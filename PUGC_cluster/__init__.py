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


# 无监督的聚类，先使用k-means
def fit_kmeans(feature_df, user_feature):
    # k-means参数设置
    # 生成2类，随机初始化质心，用不同的聚类中心初始化值运行算法的20次
    # 两类的评估系数最优
    k_means = KMeans(n_clusters=3, n_init=20, init='random', random_state=10)

    # 训练
    k_means.fit(feature_df)

    y_predict = k_means.predict(feature_df)
    # plt.scatter(feature_df[0], feature_df[1], c=y_predict)
    # plt.show()
    print(metrics.calinski_harabasz_score(feature_df, y_predict))
    # print(k_means.cluster_centers_)
    print(k_means.inertia_)
    print(metrics.silhouette_score(feature_df, y_predict))
    classif = pd.DataFrame(y_predict)

    return user_feature.join(classif)


def drowPlot():
    x_axis_data = [2, 3, 4, 5, 6]
    y_axis_data = [0.8342211351313839, 0.3696698336802127, 0.3921409057043663, 0.3046849135406272, 0.31035297867547745]

    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='SC_value')

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('clusters')
    plt.ylabel('SC')

    plt.show()


if __name__ == '__main__':
    file_path = 'data/talent_complete_columns.csv'

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

    # PCA
    #  PCA是将n维特征映射到k维上（k<n），
    #  这k维特征是全新的正交特征，称为主元，是重新构造出来的k维特征
    pca = PCA(n_components=2)

    # 降维正交后新的6维特征
    x_reduced = pca.fit_transform(feature_df)
    x_reduced = pd.DataFrame(x_reduced)

    # 合并基本的信息

    user_feature = df[['创作者ID', '创作者名称', '是否官方']]
    # print(user_feature.join(x_reduced))
    # save
    pca_feature = user_feature.join(x_reduced)
    # pca_feature.to_csv('pca_feature.csv', index=False)

    final_classif = fit_kmeans(x_reduced, user_feature)

    # final_classif.to_csv('final_classif_3.csv', index=False)
    #
    classif_cloumns = pd.merge(df, final_classif, how='inner', on='创作者ID', left_index=False, right_index=False)
    #
    print(classif_cloumns)

    classif_cloumns = pd.merge(classif_cloumns, pca_feature, how='inner', on='创作者ID', left_index=False, right_index=False)
    #
    classif_cloumns.to_csv('classif_cloumns.csv', index=False)


