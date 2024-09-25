import numpy as np


def GenerateReferencePoints(M, p):
    Zr = np.transpose(GetFixedRowSumIntegerMatrix(M, p)) / p
    return Zr

#产生参考点
def GetFixedRowSumIntegerMatrix(M, RowSum):
    #如果目标函数数量小于1
    if M < 1:
        raise ValueError('M cannot be less than 1.')

    if M != int(M):
        raise ValueError('M must be an integer.')
    # 如果目标函数数量等于1
    if M == 1:
        return np.array([RowSum])
    #创建一个A，0行M列，决策变量数
    A = np.empty((0, M))
    for i in range(RowSum + 1):
        #B是？
        B = GetFixedRowSumIntegerMatrix(M - 1, RowSum - i)
        A = np.vstack((A, np.column_stack((i * np.ones(B.shape[0]), B))))

    return A
