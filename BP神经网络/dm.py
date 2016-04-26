#!/usr/bin/python
# encoding: utf-8
import sys
import csv
import numpy as np

if __name__ == "__main__":
    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    #get reference ,that is , y(i)
    refer = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385))

    row = data.shape[0]
    col = data.shape[1]
    re = refer.reshape(row, 1)

    refermax = -10000000.0
    refermin = 1000000.0
    e = 2.71828
    print "reading finish"
    #找出最大值
    for i in range(row):
        if float(re[i][0]) > refermax:
            refermax = float(re[i][0])
        if float(re[i][0]) < refermin:
            refermin = float(re[i][0])
    #归一化
    for i in range(row):
        re[i][0] = (re[i][0] - refermin) / (refermax - refermin)


    #权重
    w = np.zeros((20, col))
    #偏倚值
    theta = list()
    otptheta = 0.2
    #学习率
    l = 0.4
    #隐藏层权重
    Midx = list()
    Midw = list()
    ErrX = list()

    #初始化权重
    for i in range(20):
        for j in range(col):
            w[i][j] = 0.1

    #初始化偏倚值
    for i in range(20):
        theta.append(0.2)
        Midw.append(0.1)
        Midx.append(0)
        ErrX.append(0)

    for k in range(10):
        print k
        for r in range(row):
            for wi in range(20):
                """
                print data[r]
                print w[wi]
                a = raw_input()
                """
                # Wij * xi
                SumWijXi = np.multiply(data[r], w[wi])

                #print SumWijXi, np.sum(SumWijXi), theta[wi][0]

                Ii = np.sum(SumWijXi) + theta[wi]
                #隐藏层X值
                Midx[wi] = 1.0 / (1 + (e)**((-1) * Ii))

                #print Midx[wi][0]

            #b =  raw_input()  

            """
            print Midx
            print Midw
            print np.multiply(Midx, Midw)
            print np.sum(np.multiply(Midx, Midw))
            """
        
            
            outputI = np.sum(np.multiply(Midx, Midw)) + otptheta

            #print outputI

            outputX = 1.0 / (1 + (e)**((-1) * outputI))

            #b =  raw_input()
            #输出层差值
            ErrOtp = outputX*(1 - outputX)*(re[r][0] - outputX)
            #输出层偏倚值更新
            otptheta += l * ErrOtp
            
            for ei in range(20):
                #隐藏层差值
                ErrX[ei] = Midx[ei] * (1 - Midx[ei]) * ErrOtp * Midw[ei]
                #隐藏层权重更新
                Midw[ei] += l * ErrOtp * Midx[ei]
                #隐藏层偏倚值更新.
                theta[ei] += l * ErrX[ei]

            for wi in range(20):
                for ci in range(col):
                    #输入层权重更新
                    w[wi][ci] += l * ErrX[wi] * data[r][ci]

            

    testdata = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    outputfile = open("output.csv", "wb")
    writer = csv.writer(outputfile)
    writer.writerow(['Id','reference'])
    resrow = 0
    for r in range(testdata):
        for wi in range(20):
            # Wij * xi
            EndSumWijXi = np.multiply(testdata[r], w[wi])
            Ii = np.sum(EndSumWijXi) + theta[wi]
            #隐藏层X值
            Midx[wi] = (1 + (e)**(-Ii))**(-1)


        outputI = np.sum(np.multiply(Midx, Midw)) + otptheta
        outputX = (1 + (e)**(-outputI))**(-1)

        writer.writerow([resrow, outputX * (refermax - refermin) + refermin])
        resrow += 1
    
    outputfile.close()  

