import numpy as np
from numba import jit, njit, prange

@njit(fastmath=True)
def selc_func(x):
    return np.log(x + 1)
    #return 1

@njit(fastmath=True)
def wy_tradegy(predmax, predmin, truemax, truemin, trueopen, trueover, setwater = 0.5):
    open_money = trueopen * (1 - setwater)
    ticket = setwater
    res = open_money
    mid = trueopen
    if truemin < mid:
        theory_ke = int((mid - predmin) / 0.01)
        real_ke = int((mid - truemin) / 0.01)
        ssum = 0.0
        for i in range(theory_ke):
            ssum += selc_func(i)
        
        for i in range(min(real_ke, theory_ke)):
            res -= ((mid - i * 0.01) * selc_func(i) / ssum) * setwater
            ticket += (selc_func(i) / ssum) * setwater


    if truemax > mid:
        theory_ke = int((predmax - mid) / 0.01)
        real_ke = int((truemax - mid) / 0.01)
        ssum = 0.0
        for i in range(theory_ke):
            ssum += selc_func(i)
        
        for i in range(min(real_ke, theory_ke)):
            res += ((mid + i * 0.01) * selc_func(i) / ssum) * setwater
            ticket -= (selc_func(i) / ssum) * setwater

    res += trueover*(ticket - setwater)
    ticket = setwater
    return res/trueover + setwater