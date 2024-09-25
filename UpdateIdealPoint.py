import numpy as np

def UpdateIdealPoint(pop, prev_zmin=None):
    if prev_zmin is None or len(prev_zmin) == 0:
        prev_zmin = np.inf * np.ones_like(pop[0]['Cost'])
        #prev_zmin = np.inf * np.ones_like(pop[0].Cost)

    zmin = np.minimum(prev_zmin, np.min([ind['Cost'] for ind in pop], axis=0))

    return zmin
