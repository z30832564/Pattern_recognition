# encoding=utf-8
import csv
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import math
def countSE(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        column = [row[0] for row in reader]
    sum1 = 0
    sum2 = 0
    for i in range(len(column)):
        sum1 = sum1+float(column[i])
    e = sum1/len(column)
    for i in range(len(column)):
        sum2 = sum2+(float(column[i])-e)**2
    s = sum2/(len(column)-1)
    return e, s

fe, fs = countSE('FEMALE.csv')
print fe, fs
me, ms = countSE('MALE.csv')
print me,ms
x = np.arange(50,300,1)
y = stats.norm.pdf(x,fe,fs)
plt.plot(x,y)
#plt.show()

def countHY(filename,fxy,mxy,fe,fs,me,ms):
    Y = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[0] for row in reader]
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column2 = [row[2] for row in reader]
    print column1
    for i in range(0, len(column1)):
        print i
        y1 = stats.norm.pdf(float(column1[i]), fe, math.sqrt(fs))   #y1表示女生类条件概率
        y2 = stats.norm.pdf(float(column1[i]), me, math.sqrt(ms))   #y2表示男生类条件概率
        p1 = (y1*fxy)/(y1*fxy+y2*mxy)
        p2 = (y2*mxy)/(y1*fxy+y2*mxy)
        print p1
        print p2
        if p1 > p2:
            Y.append('f')
        if p1 < p2:
            Y.append('m')

    q = 0   #q用来记录错误的个数
    for i in range(0, len(column2), 1):
        if Y[i] == column2[i]:
            q = q+1
        if Y[i] == column2[i].lower():
            q = q+1
    acc = float(float(q)/float((len(column2))))
    print Y,q
    print column2
    return Y, acc,q, len(column2)

Y, acc, q, r = countHY('test2.csv',0.5, 0.5, fe, fs, me, ms)
print acc
print '正确个数：', q,
print '总个数：', r
print acc

