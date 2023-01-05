import numpy as np
from numba import jit, njit, prange


@njit(fastmath=True)
def selc_func(x):
    # return np.log(x + 1)
    # return x**4
    # return 1
    return x+1.1**(-x)
    # return max(100-10*x,0)
    # return max(x-50,0.1)


@njit(fastmath=True)
def wy_tradegy(predmax, predmin, truemax, truemin, trueopen, trueover, setwater=0.5):
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


@njit(fastmath=True)
def wy_tradegy_int(predmax, predmin, truemax, truemin, trueopen, trueover, setwater=0.5, maxm=100000):
    open_money = trueopen * (1 - setwater)  # 参与交易的金额
    ticket = setwater  # 拥有股票
    res = open_money  # 用于金钱
    mid = trueopen  # 中轴
    if truemin < mid:  # 低于中轴可以买
        theory_ke = int((mid - predmin) / 0.01)  # 预测的距离刻度值 一分钱为1个刻度
        real_ke = int((mid - truemin) / 0.01)  # 实际距离
        ssum = 0.0  # selc_func归一化
        for i in range(theory_ke):
            ssum += selc_func(i)

        for i in range(min(real_ke, theory_ke)):  # 每个刻度进行铺单
            tmp = int((selc_func(i) / ssum) * setwater * maxm /
                      (mid - i * 0.01)) / (maxm / (mid - i * 0.01))  # 铺单的量
            ticket += tmp  # 买股票
            res -= (mid - i * 0.01) * tmp  # 花钱

    if truemax > mid:
        theory_ke = int((predmax - mid) / 0.01)
        real_ke = int((truemax - mid) / 0.01)
        ssum = 0.0
        for i in range(theory_ke):
            ssum += selc_func(i)

        for i in range(min(real_ke, theory_ke)):
            tmp = int((selc_func(i) / ssum) * setwater * maxm /
                      (mid + i * 0.01)) / (maxm / (mid + i * 0.01))
            ticket -= tmp
            res += (mid + i * 0.01) * tmp

    res += trueover*(ticket - setwater)
    ticket = setwater
    return res/trueover + setwater
