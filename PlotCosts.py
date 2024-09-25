import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = ['SimSun']

# 设置保存图像的文件夹路径
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')

def PlotCosts(pop,iteration):
    # 将成本值提取到列表中
    Costs = [individual['Cost'] for individual in pop]
    Costs = np.array(Costs)

    # 绘制 3D 散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Costs[:, 0], Costs[:, 1], Costs[:, 2], c='r', s=64)  # 使用 s 参数设置标记大小

    # 设置坐标轴标签
    ax.set_xlabel('总成本')
    ax.set_ylabel('碳排放')
    ax.set_zlabel('购电花费')

    # 显示网格
    ax.grid(True)

    # 显示图像
    #plt.show()

    # 创建保存目录(如果不存在)
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # 保存图像到文件
    plot_file = os.path.join(PLOT_DIR, f'plot_iteration_{iteration}.png')
    plt.savefig(plot_file)
    plt.close(fig)

