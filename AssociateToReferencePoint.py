import numpy as np

def AssociateToReferencePoint(pop, params):
    Zr = params['Zr']
    nZr = params['nZr']

    rho = np.zeros(nZr)

    d = np.zeros((len(pop), nZr))

    for i in range(len(pop)):
        for j in range(nZr):
            w = Zr[:, j] / np.linalg.norm(Zr[:, j])
            z = pop[i]['NormalizedCost']
            d[i, j] = np.linalg.norm(z - np.dot(w, z) * w)

        dmin, jmin = np.min(d[i, :]), np.argmin(d[i, :])

        pop[i]['AssociatedRef'] = jmin
        pop[i]['DistanceToAssociatedRef'] = dmin
        rho[jmin] += 1

    #print('AssociateToReferencePoint-pop', pop)

    return pop, d, rho