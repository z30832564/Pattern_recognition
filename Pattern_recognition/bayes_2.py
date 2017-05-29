# encoding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


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
    return FA
def cov(FA, flag):

    cov = np.cov(FA)
    if flag == 2:
        cov[0, 1] = 0
        cov[1, 0] = 0
    return cov

def TD(FA, cov, xy, miyu,):
    W = np.linalg.inv(cov) * (-0.5)
    w = np.linalg.inv(cov).dot(miyu)
    w1 = (miyu.T).dot(np.linalg.inv(cov)).dot(miyu) * (-0.5)
    w2 = np.log(np.linalg.det(cov)) * (-0.5)
    w3 = np.log(xy)
    w0 = w1 + w2 + w3
    g = FA.T.dot(W).dot(FA) + w.T.dot(FA) + w0
    return g



def acc(filename, xy1, xy2, flag):
    FA = readData('FEMALE.csv')
    FM = readData('MALE.csv')
    cov1 = cov(FA, flag)
    cov2 = cov(FM, flag)
    miyu1 = np.array([[np.mean(FA[0])], [np.mean(FA[1])]])
    miyu2 = np.array([[np.mean(FM[0])], [np.mean(FM[1])]])
    print cov1, miyu1
    print cov2, miyu2
    TEST = readData(filename)
    Y = []
    for i in range(0, len(TEST[0])):
        print i
        g1 = TD(np.array([TEST[0:2, i]]).T, cov1, xy1, miyu1, )
        g2 = TD(np.array([TEST[0:2, i]]).T, cov2, xy2, miyu2, )
        print g1
        print g2
        if g1 > g2:
            Y.append('f')
        if g1 < g2:
            Y.append('m')

    #计算正确率
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
    return Y, acc, q, len(y)


Y, acc, q, r = acc('test2.csv', 0.5, 0.5, flag=2)
print Y
print acc
print '正确个数：', q,
print '总个数：', r

'''
#画个图

xy = np.linspace(0, 1, 11)
print type(xy), len(xy)
acc1 = []
for i in range(0, len(xy)):
    print xy[i]
    Y, _acc, q, r = acc('test2.csv', xy[i], 1-xy[i], flag=2)
    acc1.append(_acc)
print Y
print acc1
print '正确个数：', q,
print '总个数：', r

plt.plot(xy, acc1, 'b-')
plt.show()
'''
#再画个图
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
    plot2 = pl.scatter(F1,F2,marker='.')
    plot3 = pl.scatter(M1,M2,marker='.')
    pl.legend([plot2,plot3])
    plt.show()

a = draw('test2.csv', Y)
