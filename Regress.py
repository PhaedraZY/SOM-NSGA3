from CostF import CostFunction2
from CostF import CostFunction3
from CostF import CostFunction1
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import nonzero, array
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, rand_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

'''npop=200
nVar = 7  # Number of Decision Variables
VarMin = 100  # Lower Bound of Variables
VarMax = 500  # Upper Bound of Variables
VarSize = (1, nVar)  # Size of Decision Variables Matrix'''

def TrueValue(nPop,VarMin, VarMax, VarSize):
    data = pd.DataFrame(columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'f1', 'f2', 'f3'])
    #===============生成样本点===================
    for i in range(nPop):
        u = np.random.uniform(VarMin, VarMax, VarSize)
        position = np.append(u, np.random.randint(1, 11))
        position = np.array(position)  # 将 position 转换为 NumPy 数组
        f1 = CostFunction1(position)
        f2 = CostFunction2(position)
        f3 = CostFunction3(position)
        # 将当前样本点的信息添加到 DataFrame
        data.loc[i] = [position[0], position[1], position[2], position[3],
                       position[4], position[5], position[6], f1,f2, f3]
    df = pd.DataFrame(data,index=range(nPop))
    df.to_csv('kmeans.csv', index=False)

    return df

kmeans = pd.read_csv("kmeans.csv", header=0)  # 鸢尾花数据集 Iris  class=3
df = kmeans  # 设置要读取的数据集

columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns)-3]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
attributes = len(df.columns)  # 属性数量（数据集维度）
#original_labels = list(df[columns[-1]])  # 原始标签
#print(features)

def Pseudovalues(dataset):
    # =================根据样本点聚类================
    # 输入 n 个数据对象集合Xi ;输出 k 个聚类中心 Zj 及K 个聚类数据对象集合 Cj .
    k = 5  # 聚类簇数
    T = 100  # 最大迭代数
    n = len(dataset)  # 样本数
    epsilon = 1e-5
    # 预测全部数据
    labels, centers = k_means(np.array(dataset), k, T, epsilon)
    # print(labels)
    #F_measure, ACC, NMI, RI, ARI = clustering_indicators(original_labels, labels)  # 计算聚类指标
    #print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI, "ARI", ARI)
    # print(membership)
    # print(centers)
    # print(dataset)
    draw_cluster(dataset, centers, labels=labels)

def initialize_centroids(data, k):
    # 从数据集中随机选择k个点作为初始质心
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    return centers

def get_clusters(data, centroids):
    # 计算数据点与质心之间的距离，并将数据点分配给最近的质心
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels

def update_centroids(data, cluster_labels, k):
    # 计算每个簇的新质心，即簇内数据点的均值
    new_centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def k_means(data, k, T, epsilon):
    start = time.time()  # 开始时间，计时
    # 初始化质心
    centroids = initialize_centroids(data, k)
    t = 0
    while t <= T:
        # 分配簇
        cluster_labels = get_clusters(data, centroids)

        # 更新质心
        new_centroids = update_centroids(data, cluster_labels, k)

        # 检查收敛条件
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            break
        centroids = new_centroids
        print("第", t, "次迭代")
        t += 1
    print("用时：{0}".format(time.time() - start))
    return cluster_labels, centroids


# 计算聚类指标
def clustering_indicators(labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果数据集的标签为文本类型，把文本标签转换为数字标签
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    ARI = adjusted_rand_score(labels_true, labels_pred)
    return f_measure, accuracy, normalized_mutual_information, rand_index, ARI


# 绘制聚类结果散点图
def draw_cluster(dataset, centers, labels,k):
    center_array = array(centers)
    if attributes > 2:
        dataset = PCA(n_components=2).fit_transform(dataset)  # 如果属性数量大于2，降维
        center_array = PCA(n_components=2).fit_transform(center_array)  # 如果属性数量大于2，降维
    else:
        dataset = array(dataset)
    # 做散点图
    label = array(labels)
    plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
    # plt.show()
    colors = np.array(
        ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
         "#800080", "#008080", "#444444", "#FFD700", "#008080"])
    # 循换打印k个簇，每个簇使用不同的颜色
    for i in range(k):
        plt.scatter(dataset[nonzero(label == i), 0], dataset[nonzero(label == i), 1], c=colors[i], s=7, marker='o')
    # plt.scatter(center_array[:, 0], center_array[:, 1], marker='x', color='m', s=30)  # 聚类中心
    plt.show()





