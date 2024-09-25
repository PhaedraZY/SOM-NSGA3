#====================input：输入新数据点：post[k]['position']=====================
#==========试用SOM=========
import numpy as np
import pandas as pd
from minisom import MiniSom
import pickle
from CostF import CostFunction3

def train_and_save_som(data_file, som_file):
    """
    使用 K-Means 数据建立并保存 SOM 模型

    参数:
    data_file (str): K-Means 数据文件路径
    som_file (str): 保存 SOM 模型的文件路径
    """
    # 读取数据
    df = pd.read_csv(data_file)
    features = df.columns[:7]  # 取前 7 列作为特征
    dataset = df[features].values
    print('数据集维度:', dataset.shape)

    # 创建并训练 SOM 模型
    som = MiniSom(x=5, y=5, input_len=7, sigma=1.5, learning_rate=0.5)
    som.random_weights_init(dataset)
    som.train_batch(dataset, 1000, verbose=True)


    # 获取 SOM 模型的权重,并将获胜节点坐标映射到类别标签
    weights = som.get_weights().reshape(25, 7)
    labels = {}
    for i, w in enumerate(weights):
        x, y = i // 5, i % 5
        labels[(x, y)] = i

    # 预测数据点的类别
    predicted_labels = [labels[som.winner(x)] for x in dataset]
    # 将预测的类别添加到原始数据中
    df['predicted_label'] = predicted_labels
    df.to_csv(data_file, index=False)

    # 保存 SOM 模型
    with open(som_file, 'wb') as f:
        pickle.dump(som, f)

    print("SOM 模型已保存至:", som_file)
    return som

def Pseudovalues(new_sample):
    new_sample=new_sample[:7]
    """
    使用保存的 SOM 模型预测新增样本点的类别,并找到同类最近的点

    参数:
    som_file (str): 保存 SOM 模型的文件路径
    new_sample (numpy.ndarray): 新增样本点的特征向量
    data_file (str): 包含预测结果的 K-Means 数据文件路径

    返回:
    int: 新增样本点的类别标签
    int: 同类最近的点在数据中的索引
    float: 同类最近的点的 f3 特征值
    """
    # 加载 SOM 模型
    with open("som_model.pkl", 'rb') as f:
        som = pickle.load(f)

    # 获取新样本点的获胜节点
    new_winner = som.winner(new_sample)

    # 将获胜节点坐标映射到类别标签
    weights = som.get_weights().reshape(25, 7)
    labels = {}
    for i, w in enumerate(weights):
        x, y = i // 5, i % 5
        labels[(x, y)] = i

    new_label = labels[new_winner]

    # 读取 K-Means 数据,并找到同类最近的点
    df = pd.read_csv("kmeans.csv")
    features = df.columns[:7]
    all_row_points=df[df['predicted_label'] == new_label]
    same_class_points = df[df['predicted_label'] == new_label][features]
    distances = np.linalg.norm(same_class_points.values - new_sample, axis=1)
    nearest_index = np.argmin(distances)
    nearest_point = all_row_points.iloc[nearest_index].values
    #print("新样本点的类别:", new_label,"同类最近的点在数据中的索引:", nearest_index)
    #print('执行方法：Pseudovalues')
    return nearest_point[10]

def PseudoOrTrueValue(it,new_pop):
    if it % 5 == 0:
        f3 = CostFunction3(new_pop)
    else:
        f3 = Pseudovalues(new_pop)
    return f3

# 训练并保存 SOM 模型
train_and_save_som("kmeans.csv", "som_model.pkl")




