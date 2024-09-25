import numpy as np
# 交叉操作
def Crossover(x1,x2):
    alpha = np.random.rand(*x1.shape)

    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1

    return y1,y2