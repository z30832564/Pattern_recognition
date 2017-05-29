# encoding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

#读取文件成矩阵形式，文件的前两列为矩阵的前两列
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

#读取男生女生数据
FA = readData('FEMALE.csv')
FM = readData('MALE.csv')
print FA
print type(FA)

#求类均值向量
def M(x):
    n = len(x)
    M = np.zeros(len(x.T))
    for i in range(0, n-1):
        m = x[i]
        M = M+m
    M = M/n
    return np.array([M]).T

m1 = M(FA)
m2 = M(FM)
print m1
print m2

#求类内离散度矩阵
def S(x, m):
    S = np.zeros([len(x[0]), len(x[0])])
    for i in x:
        s = (np.array(i).T-m).dot(np.array(i).T-m).T
        S = S+s
    return s

s1 =S(FA, m1)
s2 = S(FM, m2)
print s1
sw = s1 + s2
print sw

w = np.linalg.inv(sw).dot(m1-m2)

print w


def JC(filename, m1, m2, xy1, xy2, w):
    Y = []
    x = readData(filename)
    mu1 = w.T.dot(FA.T).mean()
    mu2 = w.T.dot(FM.T).mean()
    b = (len(FA[0])*mu1+len(FM[0])*mu2)/(len(FM[0])+len(FA[0]))
    print b
    for i in x:
        '''
        g = w.T.dot((np.array([i]).T) - ((m1+m2)*0.5))-(np.log(xy2/xy1))
        print g
        if g<0:
            Y.append('f')
        else:
            Y.append('m')
        '''
        g = w.T.dot(np.array([i]).T)

        if g < b:
            Y.append('f')
        else:
            Y.append('m')
    # 计算正确率
    q = 0
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        y = [row[2] for row in reader]
    for i in range(0, len(y), 1):
        if Y[i] == y[i]:
            q = q + 1
        if Y[i] == y[i].lower():
            q = q + 1
    if q > len(y):
        q = q / 2
    acc = float(float(q) / float((len(y))))
    return Y, acc, q, len(y), b

def draw(filename):
    x1 = range(150, 200, 10)
    l = (b-(w[0]*x1))/(w[1])
    plot1 = pl.plot(x1,l)
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
    for i in range(0, len(column3)):
        if column3[i] == 'f':
            F1.append(column1[i])
            F2.append(column2[i])
        if column3[i] == 'm':
            M1.append(column1[i])
            M2.append(column2[i])
        if column3[i] == 'F':
            F1.append(column1[i])
            F2.append(column2[i])
        if column3[i] == 'M':
            M1.append(column1[i])
            M2.append(column2[i])
    plot2 = pl.scatter(F1,F2,marker='.')
    plot3 = pl.scatter(M1,M2,marker='.')
    pl.legend([plot1,plot2,plot3])
    plt.show()


Y, acc, q, r, b = JC('test1.csv', m1, m2, 0.5, 0.5, w)
print Y
print acc
print '正确个数：', q
print '总个数：', r
print w[0], w[1]

draw('test1.csv')
