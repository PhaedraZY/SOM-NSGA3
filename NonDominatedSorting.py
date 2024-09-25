from Dominates import Dominates

def NonDominatedSorting(pop):
    nPop = len(pop)

    for i in range(nPop):
        pop[i]['DominationSet'] = []
        pop[i]['DominatedCount'] = 0

    F = [[]]

    for i in range(nPop):
        for j in range(i + 1, nPop):
            if Dominates(pop[i], pop[j]):
                pop[i]['DominationSet'].append(j)
                pop[j]['DominatedCount'] += 1
            if Dominates(pop[j], pop[i]):
                pop[j]['DominationSet'].append(i)
                pop[i]['DominatedCount'] += 1


        if pop[i]['DominatedCount'] == 0:
            F[0].append(i)
            pop[i]['Rank'] = 1

    k = 0
    while True:
        Q = []
        for i in F[k]:
            p=pop[i]
            for j in pop[i]['DominationSet']:
                q=pop[j]
                q['DominatedCount'] -= 1
                if q['DominatedCount'] == 0:
                    Q.append(j)
                    q['Rank'] = k + 2
                pop[j]=q
        if not Q:
            break
        F.append(Q)
        k += 1

    return pop, F
