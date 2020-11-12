import math
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.neighbors import kde
import scipy.stats as sta
from openpyxl import Workbook
import pandas as pd
from functools import reduce


# 使用高斯核函数计算（自动计算带宽）


def gaussian1(x):
    """

    一维的高斯估计
    x -- 一维数组
    return: gaussian 估计密度值
    """
    re = sta.gaussian_kde(x)
    return re(x)


def gaussian2(x, y):
    """
    二维的高斯估计
    x ,y-- 一维数组
    return: gaussian 估计密度值
    """
    rt = np.vstack([x, y])
    re = sta.gaussian_kde(rt)
    return re(rt)


# 计算带宽（使用kde计算时使用）


def h1decision(b):
    """
    计算带宽
    :param b:一维数组  -- ndarray
    :return: 返回这组数据带宽-- int
    """
    # 这里使用样本标准差
    if len(b) == 1:
        return 1.05 * 0.08
    a1 = np.std(b, ddof=1)
    l = len(b)
    tmp = math.pow(l, -0.2)
    res1 = 1.05 * a1 * tmp
    return res1
    # log = kde.KernelDensity(kernel='gaussian', bandwidth=res).fit(r).score_samples(r)


def h2decision(a, b):
    """
    二维变量的带宽处理
    :param a: 一维数组  -- ndarray
    :param b: 一维数组  -- ndarray
    :return: 二维数据的带宽
    """
    tmp = np.concatenate((a, b), axis=0)
    st = np.std(tmp, ddof=1)
    l1 = len(a) + len(b)  # 两组数据总长
    tmp1 = math.pow(l1, -0.2)
    res1 = 1.05 * st * tmp1
    # print("res:", res, "res1: ", res1)
    return res1


# 一些格式的转换


def dTl(x):
    """
    dataframe转为list
    :param x: DataFrame
    :return: 一维数组
    """
    temp = x.values
    res = temp.flatten()  # 降维
    return res


def lT2n(x):
    """
    list转二维数组
    x -- list
    return ： 二维数组
    """
    a1 = np.array(x)
    # print("a1",type(a1),a1)
    re = a1.reshape((len(a1), 1))
    return re


# 将2维数组以一列为单位存入excel表中


def format_excel(name, feature):
    """
    将传入数组以列形式传入excel
    :param feature: 具体内容 -- ndarray  二维数组
    :param name: String 为sheet的名字
            # label:第0行的标签  --ndarray 二维数组
    :return: wb - >在主函数中再
    使用wb.save("名字.xlsx")
    """
    wb = Workbook()
    ws = wb.create_sheet(name)

    # label_input = []
    # for l in range(len(label)):
    #     label_input.append(label[l][0])
    # ws.append(label_input) #标签

    # dex = len(feature[0])
    # for i in range(1,len(feature)):
    #     if len(feature[i])> dex:
    #         dex = len(feature[i])

    for f in range(len(feature[0])):
        ws.append(feature[:, f].tolist())  # 写入一行
    return wb


# 2维核密度kde求解


def kde2D(x, y, bandwidth):
    """
    kde计算联合概率
    :param x: 一维数组x
    :param y: y
    :param bandwidth:
    :return: 联合概率 -- np.darray
    """
    # xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T
    # vstack--竖直堆叠序列中的数组（行方向），使两个数组成为一行
    # print("1:" ,np.vstack([y, x]),type(np.vstack([y, x])))
    # print("2:" ,xy_train,type(xy_train))
    # .T使得两数组y，x一一对应形成一个个数组length（x）
    # print(xy_train)
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(xy_train)
    log_P = kde_skl.score_samples(xy_train)
    P = np.exp(kde_skl.score_samples(xy_train))
    # print("log_P:",log_P)
    # print("P :",P)
    return P


# 0-1化结果


def minmax(x1):
    '''
    最大最小值标准化【0，1】
    :param x1: 需要标准化的二维数组ndarray
    :return: res :标准化后的一维数组ndarray
    '''
    x = x1.reshape(len(x1), 1)
    min_max_scale = preprocessing.MinMaxScaler(feature_range=(0.00000000001, 1))
    tmp = min_max_scale.fit_transform(x)
    res = tmp.flatten()
    return res


# 离散化--MDLP


# 定义信息熵计算
def ent(x):
    """
    :param x: 其中输入的x为一列数据
    :return: 该列数据的信息熵
    """
    x = pd.Series(x)
    # 定义初始信息熵ent1
    ent1 = 0
    p = x.value_counts() / len(x)
    for i in range(len(p)):
        e = -p.iloc[i] * math.log(p.iloc[i])
        ent1 = ent1 + e
    return ent1


# 定义计算切点函数：切分数据的切点以及切分后的信息熵
def cutIndex(x, y):
    """
    :param x: 其中输入的x为一列数据
    :param y: 标签数据
    :return: 该列数据的切点以及划分后的信息熵
    """
    n = len(x)
    x = pd.Series(x)
    # 初始化一个熵的值
    entropy = 9999
    cutD = None
    # 寻找最佳分裂点
    for i in range(n - 1):
        if x.iloc[i + 1] != x.iloc[i]:
            # cutX=(x.iloc[i+1]+x.iloc[i])/2
            wCutX = x[x < x.iloc[i + 1]]
            wn = len(wCutX) / n
            # 左边权重wn
            e1 = wn * ent(y[:len(wCutX)])
            # 右边权重
            e2 = (1 - wn) * ent(y[len(wCutX):])
            # 权重总和的信息熵
            val = e1 + e2
            if val < entropy:
                entropy = val
                cutD = i
    if cutD is None:
        return None
    else:
        # 返回切点,最小信息熵
        return (cutD, entropy)


# 定义停止切分函数：根据切分后信息熵的变化来制定停止切分条件
def cutStop(cutD, y, entropy):
    '''
    :param cutD: 切点
    :param y: 标签列
    :param entropy:信息熵
    :return: 信息熵的变化
    '''
    n = len(y)
    es = ent(y)  # 总信息熵
    gain = es - entropy  # 信息熵的变化
    left = len(set(y[0:cutD]))
    right = len(set(y[cutD:]))
    lengthY = len(set(y))
    if (cutD == None or lengthY == 0):
        return (None)
    else:
        # math.log(3^2-2) 也就是math.log(7)
        delta = math.log(3 ** lengthY - 2) - (lengthY * ent(y) - left * ent(y[0:cutD]) - right * ent(y[cutD:]))
        cond = math.log(n - 1) / n + delta / n
        # 大约选择变化小于总体信息熵的20%(例如:总体熵为1,当切点且分后变化小于0.2的时候不需要进行切分)
        if (gain < cond):
            return (None)
        else:
            return (gain)


# 定义切分函数：通过递归的方式切分数据且返回所有符合切点的位置
def cutPoints(x, y):
    """
    :param x: 输入带切分的列数据
    :param y: 输入标签数据
    :return: 返回切点
    """
    dx = x.sort_values()  # 默认按数据大小升序排序
    dy = pd.Series(y, index=dx.index)  # 按照X的排序更新y的值
    depth = 0

    def gr(low, upp, depth=depth):
        x = dx[low:upp]
        y = dy[low:upp]
        n = len(y)
        k = cutIndex(x, y)
        if k is None:
            return None  # 判断是否存在切点 加权求和
        else:
            cutD = k[0]  # 切点位置
            entropy = k[1]  # 信息熵
            gain = cutStop(cutD, y, entropy)
            if gain is None:
                return None  # 判断是否应该切分(基于熵的变化)
            else:
                return [cutD, depth + 1]

    # 递归函数返
    def part(low=0, upp=len(dx), cutTd1=[], depth=depth):
        x1 = dx[low:upp]
        y1 = dy[low:upp]
        n = len(x1)
        # 返回的是切点 与depth+1
        k = gr(low, upp, depth=depth)
        if n < 2 or k is None:
            return cutTd1
        else:
            cutX = k[0]
            depth = depth + 1
            cutTd1.append(low + cutX)
            # cutTd1.extend([low+cutX])
            # 从小到大
            cutTd1.sort()
            return part(low, low + cutX, cutTd1, depth) + part(cutX + low, upp, cutTd1, depth)

    res1 = part(low=0, upp=len(dx), cutTd1=[], depth=depth)
    cutDx = []
    if not res1:
        return None
    # 去重
    func = lambda x, y: x if y in x else x + [y]
    res = reduce(func, [[], ] + res1)
    res = pd.Series(res)
    # 返回切点的label 对应的实际值
    for i in res.values:
        k = round((x.sort_values().values[i] + x.sort_values().values[i + 1]) / 2, 6)
        cutDx.append(k)
    return cutDx


# Mdlpx 通过前述函数返回最适合的切点列表，返回有两个部分
def Mdlp(data):
    """
    :param data: 输入的是df 需要带有label字段
    :return: [切点] [min,切点,max]
    """
    p = data.shape[1] - 1  # 直接用.shape可以快速读取矩阵的形状
    y = data.iloc[:, p]  # 取数据
    # xd=data
    cutP = []
    cutPs = []
    for i in range(0, p):
        x = data.iloc[:, i]
        cuts1 = cutPoints(x, y)  # x是一列的数据，y是标签
        # cuts1是切点
        if cuts1 is None:
            cuts1 = "ALL"
        cuts = [[min(x)], cuts1, [max(x)]]
        cutPs.append(cuts)
        cutP.append(cuts1)
        # label=range(1,len(cuts))
        # xd.ix[:,i]=pd.cut(x,bins=cutpc,labels=label,include_lowest=True)
    return cutP, cutPs


def Mdlpx(data):
    """
    :param data: 输入的是df 需要带有label字段
    :return:  {column_name:切点} {column_name:min,切点,max}
    """
    cut, cutAll = Mdlp(data)
    ziDian = {}
    ziDian_all = {}
    for i in range(data.shape[1] - 1):
        ziDian[data.columns[i]] = cut[i]
        ziDian_all[data.columns[i]] = cutAll[i]
    return ziDian, ziDian_all


