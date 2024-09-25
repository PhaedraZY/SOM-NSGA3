import numpy as np
from NormalizePopulation import NormalizePopulation
from NonDominatedSorting import NonDominatedSorting
from AssociateToReferencePoint import AssociateToReferencePoint

#==============
def SortAndSelectPopulation(pop, params):
    #归一化
    pop, params = NormalizePopulation(pop, params)
    #print('NormalizePopulation-pop',pop)
    pop, F = NonDominatedSorting(pop)
    #print('NonDominatedSorting', pop)
    #F帕累托前沿

    nPop = params['nPop']
    if len(pop) == nPop:
        return pop, F, params

    pop, d, rho = AssociateToReferencePoint(pop, params)

    newpop = []
    LastFront = []
    for l in range(len(F)):
        if len(newpop) + len(F[l]) > nPop:
            LastFront = F[l]
            break
        newpop.extend([pop[i] for i in F[l]])

    while True:
        j = np.argmin(rho)

        AssocitedFromLastFront = []
        for i in LastFront:
            if pop[i]['AssociatedRef'] == j:
                AssocitedFromLastFront.append(i)

        if not AssocitedFromLastFront:
            rho[j] = np.inf
            continue
        #print('AssocitedFromLastFront',AssocitedFromLastFront)

        if rho[j] == 0:
            ddj = d[AssocitedFromLastFront, j]
            new_member_ind = np.argmin(ddj)
            #print('0 new_member_ind',new_member_ind)
        else:
            new_member_ind = np.random.choice(AssocitedFromLastFront)
            #print('1 new_member_ind', new_member_ind)
        #=====================================================
        #MemberToAddIndex = AssocitedFromLastFront.index(new_member_ind)
        LastFront.remove(new_member_ind)
        newpop.append(pop[new_member_ind])

        rho[j] += 1

        if len(newpop) >= nPop:
            break

    pop, F = NonDominatedSorting(newpop)

    return pop, F, params
