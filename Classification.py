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

def TrueVal(nPop, VarMin, VarMax, VarSize):
    data = []
    for i in range(nPop):
        u = np.random.uniform(VarMin, VarMax, VarSize)
        position = np.append(u, np.random.randint(1, 11))
        position1 = np.array(position)  # 将 position 转换为 NumPy 数组
        f1 = CostFunction1(position1)
        f2 = CostFunction2(position1)
        f3 = CostFunction3(position1)

        # 将数据添加到 data 列表中
        data.append({
            'pos_1': position1[0], 'pos_2': position1[1], 'pos_3': position1[2], 'pos_4': position1[3],
            'pos_5': position1[4], 'pos_6': position1[5], 'pos_7': position1[6], 'pos_8': position1[7],
            'f1': f1, 'f2': f2, 'f3': f3
        })

    # 创建 DataFrame 并保存到 CSV 文件
    df = pd.DataFrame(data, index=range(nPop))
    df.to_csv('kmeans.csv', index=False)

    return position

position=TrueVal(200,100,500,7)
print(position)
'''kmeans = pd.read_csv("kmeans.csv", header=0)  #数据集 Iris  class=5
df = kmeans  # 设置要读取的数据集
columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns)]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
attributes = len(df.columns)  # 属性数量（数据集维度）
#original_labels = list(df[columns[-1]])  # 原始标签'''


def Classfier(dataset):
    # 将数据集划分为自变量和目标函数值
    X = dataset.iloc[:, :-1]  # 前7个维度作为自变量
    y = dataset.iloc[:, -3:]  # 后3个维度作为目标函数值

    # =================根据样本点聚类================
    n_clusters = 5  # 设置聚类数量为 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)

    # ==========获取聚类结果============
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("聚类标签:")
    print(labels)
    print("质心:")
    print(centroids)

    return labels,centroids,kmeans

def Pseudovalues(new_point):
    labels,centroids,kmeans = Classfier(dataset)
    # 将数据集划分为自变量和目标函数值
    X = dataset.iloc[:, :-1]  # 前7个维度作为自变量
    y = dataset.iloc[:, -3:]  # 后3个维度作为目标函数值
    # =============预测新数据点的类别===========
    predicted_cluster = kmeans.predict([new_point])
    # 计算新数据点到各个聚类中心的距离
    distances = kmeans.transform([new_point])
    print("新数据点属于类别:", predicted_cluster[0])

    # ==========找到新数据点所属类中最近的点=======
    cluster_indices = np.where(labels == predicted_cluster[0])[0]
    cluster_data = X.iloc[cluster_indices]
    cluster_targets = y.iloc[cluster_indices]
    closest_point_index = np.argmin(distances[0, predicted_cluster[0]])
    closest_point = cluster_data.iloc[closest_point_index]
    closest_target = cluster_targets.iloc[closest_point_index]

    print("新数据点所属类中最近的点:", closest_point)
    print("新数据点的目标函数值:", closest_target)
    # 获取 closest_target 的第三个元素(f3)
    closest_f3 = closest_target.iloc[2]
    return closest_f3

#p=Pseudovalues(np.random.rand(7))
#print(p)







