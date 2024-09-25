import numpy as np

# =========标量化==========
def GetScalarizingVector(nObj, j):
    # 目标数量 nObj 和当前目标的索引 j
    epsilon = 1e-10
    w = epsilon * np.ones(nObj)
    w[j] = 1
    # 它返回一个长度为 nObj 的权重向量 w,其中除了第 j 个元素为 1,其他元素都为一个很小的值 epsilon
    return w

def PerformScalarizing(z, params):
    # 目标值矩阵z和一个包含相关参数的字典params
    # 传过来的参数：fp，params
    # 更新params的参数：zmax，smin
    nObj, nPop = z.shape

    if params['smin'] is None:
        zmax = params['zmax']
        smin = params['smin']
        #print('smin',smin)
    else:
        zmax = np.zeros((nObj, nObj))
        smin = np.full(nObj, np.inf)


    for j in range(nObj):
        # w是每个目标的权重向量，nobj是它的长度，除了第j个元素为1，其他元素都为一个很小的值
        w = GetScalarizingVector(nObj, j)
        # 加权切比雪夫标量化值s，z / w[:, None]得最大值
        # z是一个nobj x npop的矩阵，表示种群在每个目标函数上的值
        # w[:, None]将权重向量w扩展为一个nobj x 1的矩阵，与z（fp）进行运算
        # z / w[:, None]：nObj × nPop，表示每个个体在每个目标上的加权值===========不同个体，的目标函数，加权值不同
        #print('z / w[:, None]:', z / w[:, None])
        s = np.max(z / w[:, None], axis=0)
        # print('s', s)    # s没有0
        # s为一个长度为nPOP的向量，表示每个个体的最大加权值——找到=每个=个体的最大加权值，沿着目标维度找
        # s是目标函数*w之后的加权值！

        sminj = np.min(s)
        # 找到s的最小值，即表示第j个目标的最小标量值
        # print('sminj',sminj)
        if sminj < smin[j]:
            zmax[:, j] = z[:, np.argmin(s)]
            # np.argmin(s) 找到 s 中最小值的索引,找到z（fp）中对应的目标值向量
            smin[j] = sminj
        # 更新zmax，更新为s最小值对应的目标值向量
        # 更新smin：将smin【j】更新为当前目标的最小值标量sminj

    params['zmax'] = zmax
    params['smin'] = smin
    # 这个函数的目的是找到当前个体中的最小值。更新个体的位置
    return params

