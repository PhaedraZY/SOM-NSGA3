import numpy as np

import numpy as np
from UpdateIdealPoint import UpdateIdealPoint
from PerformScalarizing import PerformScalarizing

def FindHyperplaneIntercepts(zmax):
    # 这个函数 FindHyperplaneIntercepts(zmax) 的作用是计算一个超平面在每个维度上的截距。
    # 函数的输入 zmax 是一个二维数组,其中每一行表示一个数据点,每一列表示一个特征维度。
    #aa= np.ones((1, zmax.shape[1]))
    #bb = zmax
    #w = np.linalg.solve(zmax.T, np.ones((1, zmax.shape[1])).T).T

    w = np.ones((1, zmax.shape[1])) @ np.linalg.pinv(zmax)
    a = (1 / w).T
    return a

def NormalizePopulation(pop, params):
    params['zmin'] = UpdateIdealPoint(pop, params['zmin'])  # 已标准化
    # print('UpdateIdealPoint:zmin:', params['zmin'])
    # 得到理想点得值
    # 每个目标与理想点之间的差距
    fp = np.array([ind['Cost'] for ind in pop]) - np.tile(params['zmin'], (len(pop), 1))
    fp = fp.T    # fp 3x200 ndarray 3是目标函数的数量 100是个体数：npop fp不是目标函数，而是差距
    #print('# 每个目标与理想点之间的差距fp：', fp)

    # 获取当前zmax smin
    params = PerformScalarizing(fp, params)
    #print('params[zmax]:',params['zmax'])


    a = FindHyperplaneIntercepts(params['zmax'])
    #print('a', a)

    for i in range(len(pop)):
        aa = fp[:, i].reshape((3, 1))
        bb = a.reshape((3,1))
        #pop[i]['NormalizedCost'] = fp[:, i].T / a
        pop[i]['NormalizedCost'] = (aa / bb).flatten()
        #print('pop[i][NormalizedCost]', pop[i]['NormalizedCost'])

    return pop, params

