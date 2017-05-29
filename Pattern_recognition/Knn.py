# encoding=utf-8
import numpy as np
import csv
import pylab as pl
import matplotlib.pyplot as plt
import random
def readData(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[0] for row in reader]
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column2 = [row[1] for row in reader]
    fa1 = np.array(column1)
    fa11 = fa1.astype(float)
    fa2 = np.array(column2)
    fa22 = fa2.astype(float)
    FA = np.array([fa11, fa22])
    FA = FA.T
    return FA
FA = readData('FEMALE.csv')
FM = readData('MALE.csv')
print FA
def cos(vector1,vector2):
    dot_product = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += (a-b)**2
    if dot_product == 0.0:
        return None
    else:
        return (dot_product)**0.5


def knn(FA, FM, x, N):
    Distance = []
    index = []
    F = 0
    M = 0
    if N>1:
        for i in range(0,len(FA)):
            d = cos(x, FA[i])
            Distance.append(d)
        for i in range(0, len(FM)):
            d = cos(x, FM[i])
            Distance.append(d)
        for i in range(0, N):
            index_v = Distance.index(min(Distance))
            Distance[Distance.index(min(Distance))] = 10
        index.append(index_v)
    else:
        for i in range(0,len(FA)):
            d = cos(x, FA[i])
            Distance.append(d)
        for i in range(0, len(FM)):
            d = cos(x, FM[i])
            Distance.append(d)
        index_v = Distance.index(min(Distance))
        index.append(index_v)
    for i in index:
        if i <= (len(FA)-1):
            F = F+1
        else:
            M = M + 1
    if F>M:
        return 'f'
    else:
        return 'm'

def Acc(filename, Y):
    q = 0
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        y = [row[2] for row in reader]
    for i in range(0, len(y)-1, 1):
        if Y[i] == y[i]:
            q = q + 1
        if Y[i] == y[i].lower():
            q = q + 1
    if q > len(y):
        q = q / 2
    acc = float(float(q) / float((len(y))))
    return acc

def classfy(filename,N):
    Y = []
    test = readData(filename)
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[0] for row in reader]
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column2 = [row[1] for row in reader]
    for i in range(0, len(column2)):
        x = [float(column1[i]), float(column2[i])]
        Y.append(knn(FA, FM, x, N))
    acc = Acc(filename, Y)
    return Y, acc


def YASUO():
    FA = readData('FEMALE.csv')
    FM = readData('MALE.csv')
    Store = []
    Grabbag = []
    index_ = []
    for i in range(0, len(FA)):
        Grabbag.append(FA[i])
    for i in range(0, len(FM)):
        Grabbag.append(FM[i])
    print Grabbag, len(Grabbag)
    #初始化
    Store.append(Grabbag[0])
    index_.append(0)

    while(1):
        for i in range(1, len(Grabbag)):
            print '迭代', i
            Distance = []
            count = 0
            for j in range(0, len(Store)):
                d = cos(Grabbag[i], Store[j])
                Distance.append(d)
            print Distance
            min_ = min(Distance)
            print min_
            index_v = index_[Distance.index(min_)]
            print index_v
            if index_v <= (len(FA)-1) and i <= (len(FA)-1):
                flag = 1
            elif index_v > (len(FA)-1) and i > (len(FA)-1):
                flag = 1
            else:
                flag = 0
            if flag == 0:
                Store.append(Grabbag[i])
                index_.append(i)
                count = count+1
        if count == 0:
            break
        if len(Store) == len(Grabbag):
            break
    FA_ = []
    FM_ = []
    for m in range(0, len(index_)):
        if index_[m] <= (len(FA)-1):
            FA_.append(Grabbag[index_[m]])
        else:
            FM_.append(Grabbag[index_[m]])
    return FA_, FM_


FA, FM = YASUO()
print FA, len(FA)
print FM, len(FM)
Y, acc = classfy('test2.csv', 1)
print Y
print acc


def draw(filename, Y):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[0] for row in reader]
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column2 = [row[1] for row in reader]
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column3 = [row[2] for row in reader]
    F1 = []
    F2 = []
    M1 = []
    M2 = []
    print len(Y)
    print len(column3)
    for i in range(0, len(Y)):
        if Y[i] == 'f':
            F1.append(column1[i])
            F2.append(column2[i])
        if Y[i] == 'm':
            M1.append(column1[i])
            M2.append(column2[i])
        if Y[i] == 'F':
            F1.append(column1[i])
            F2.append(column2[i])
        if Y[i] == 'M':
            M1.append(column1[i])
            M2.append(column2[i])
    plot2 = pl.scatter(F1, F2,marker='.')
    plot3 = pl.scatter(M1, M2,marker='.')
    pl.legend([plot2,plot3])
    plt.show()

a = draw('test2.csv', Y)