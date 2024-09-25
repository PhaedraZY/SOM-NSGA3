import numpy as np
from MOP2 import CostFunction
from GenerateReferencePoints import GenerateReferencePoints
from Crossover import Crossover
from Mutate import Mutate
from SortAndSelectPopulation import SortAndSelectPopulation
from PlotCosts import PlotCosts
from numpy.random import randint
import pandas as pd
from CostF import CostFunction1
from CostF import CostFunction2
from CostF import CostFunction3
from Pseudovalues import Pseudovalues
from Pseudovalues import PseudoOrTrueValue
import threading
import time

# 记录开始时间
start_time = time.time()
# ===============Problem Definition=====================
nVar = 7  # Number of Decision Variables
VarMin = 100  # Lower Bound of Variables
VarMax = 500  # Upper Bound of Variables
VarSize = (1, nVar)  # Size of Decision Variables Matrix
u = np.random.uniform(VarMin, VarMax, VarSize)
print(u)

# 用于得到目标函数的数量
nObj = len(CostFunction(np.append(u, randint(1, 11))))

# NSGA-III 参数
nDivision = 10
Zr = GenerateReferencePoints(nObj, nDivision)#输入目标数量、决策变量数量

MaxIt = 10  # Maximum Number of Iterations
nPop = 200  # Population Size
pCrossover = 0.5  # Crossover Percentage
nCrossover = 2 * round(pCrossover * nPop / 2)  # Number of Parents (Offsprings)
pMutation = 0.5  # Mutation Percentage
nMutation = round(pMutation * nPop)  # Number of Mutants
mu = 0.02  # Mutation Rate
sigma = 0.1 * (VarMax - VarMin)  # Mutation Step Size

# Initialization
print('Starting NSGA-III ...')

# 定义 params 为字典
params = {
    'nPop': nPop,
    'Zr': Zr,
    'nZr': Zr.shape[1],
    'zmin': [],
    'zmax': [],
    'smin': []
}

# 定义 empty_individual 为字典
empty_individual = {
    'Position': [],
    'Cost': [],
    'Rank': [],
    'DominationSet': [],
    'DominatedCount': [],
    'NormalizedCost': [],
    'AssociatedRef': [],
    'DistanceToAssociatedRef': []
}

#定义种群
# 创建 pop DataFrame
pop = [empty_individual.copy() for _ in range(nPop)]

#采样-均匀分布
from scipy.stats import qmc

# 创建拉丁超立方采样器
sampler = qmc.LatinHypercube(d=VarSize[1])

#=================生成初始种群=======================
    #u = np.random.uniform(VarMin, VarMax, VarSize)
    # 生成拉丁超立方采样
    #u = sampler.random(n=VarSize[0])
    # 将采样值缩放到 [VarMin, VarMax] 范围内 
    #u = VarMin + u * (VarMax - VarMin)
# ==========c初始化==============
for i in range(nPop):
    u = np.random.uniform(VarMin, VarMax, VarSize)
    pop[i]['Position'] = np.append(u, np.random.randint(1, 11))
    pop[i]['Cost'] = CostFunction(pop[i]['Position'])
    #将第三个函数赋值为伪值
    #print(pop[i]['Cost'])

# ++0+++0+++0+++0++第二处修改+++0+++0+++0+++0+
pop, F, params = SortAndSelectPopulation(pop, params)

# ====================NSGA=====================
# ==============从迭代时开始：用阁几代伪值、隔几代真实值的方法进行评估===============
# NSGA-III Main Loop
for it in range(1, MaxIt + 1):
    # ====================Crossover===================
    #popc = [empty_individual() for _ in range(nCrossover)]
    popc = [empty_individual.copy() for _ in range(nCrossover)]
    #popc = [pd.DataFrame(empty_individual) for _ in range(nCrossover)]

    for k in range(0, nCrossover, 2):
        i1 = randint(0, nPop)
        p1 = pop[i1]

        i2 = randint(0, nPop)
        p2 = pop[i2]

        popc[k]['Position'], popc[k + 1]['Position'] = Crossover(p1['Position'], p2['Position'])
        popc[k]['Cost'] = [CostFunction1(popc[k]['Position']),CostFunction2(popc[k]['Position']),PseudoOrTrueValue(it,popc[k]['Position'])]
        popc[k + 1]['Cost'] = [CostFunction1(popc[k+1]['Position']),CostFunction2(popc[k+1]['Position']),PseudoOrTrueValue(it,popc[k+1]['Position'])]

    popc = popc
    #print('popc',popc)

    # =====================Mutation=====================
    #popm = [empty_individual() for _ in range(nMutation)]
    #popm = [pd.DataFrame(empty_individual) for _ in range(nMutation)]
    popm = [empty_individual.copy() for _ in range(nMutation)]
    for k in range(nMutation):
        i = randint(0, nPop)
        p = pop[i]
        popm[k]['Position'] = Mutate(p['Position'], mu, sigma)
        popm[k]['Cost'] = [CostFunction1(popm[k]['Position']),CostFunction2(popm[k]['Position']),PseudoOrTrueValue(it,popm[k]['Position'])]

    # =======================Merge=======================
    pop = pop + popc + popm

    # =======Sort Population and Perform Selection========
    pop, F, params = SortAndSelectPopulation(pop, params)

    # ===============Store F1帕累托前沿=====================
    F1 = [pop[i] for i in F[0]]

    # Show Iteration Information
    print(f'Iteration {it}: Number of F1 Members = {len(F1)}')

    # Plot F1 Costs
    PlotCosts(F1,it)

# 你的 NSGA-III 算法代码

# 记录结束时间并计算总执行时间
end_time = time.time()
total_time = end_time - start_time

# 打印总执行时间
print(f"Total execution time: {total_time:.2f} seconds")

import pandas as pd
pop_df = pd.DataFrame(pop)
f1_df = pd.DataFrame([ind for ind in F1])
pop_df.to_csv('pop.csv')
f1_df.to_csv('f1.csv')
params_df = pd.DataFrame(params)
params_df.to_csv('params.csv')


print(f'Final Iteration: Number of F1 Members = {len(F1)}')
print('Optimization Terminated.')


