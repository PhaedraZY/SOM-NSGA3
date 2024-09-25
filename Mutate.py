import numpy as np
#变异操作：
def Mutate(x,mu,sigma):
    nVar = len(x)
    nMu = np.ceil(mu * nVar).astype(int)
    j = np.random.choice(nVar, nMu, replace=False)
    #随便选取j位置的元素进行编译操作
    y = np.copy(x)
    y[j] = x[j] + sigma * np.random.randn(nMu)
    return y