import numpy as np
def Dominates(x, y):
    return (np.array(x['Cost']) <= np.array(y['Cost'])).all() and (np.array(x['Cost']) <= np.array(y['Cost'])).any()

