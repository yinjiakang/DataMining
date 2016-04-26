#!/usr/bin/python
# encoding: utf-8
import sys
import csv
import numpy as np

if __name__ == "__main__":

    #get data
    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    #get reference ,that is , y(i)
    refer = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385))

    alpha = 0.005
    row = data.shape[0]
    re = refer.reshape(row, 1)

    # theta0 ----  theta384
    theta = np.zeros((data.shape[1], 1))
    data[:, 0] = 1

    for k in range(50000):
        # h(x)
        MaxProduct = np.dot(data, theta)
        # h(x) - y
        HxMinusY = np.subtract(MaxProduct, re)
        # (h(x) - y) * Xi
        MultiplyXi = np.multiply(HxMinusY, data)
        # sum of (h(x) - y) * Xi
        Sum = np.sum(MultiplyXi, axis = 0).reshape(data.shape[1], 1)
        # alpha / m * (h(x) - y) * Xi
        LaterItems = np.multiply(Sum, alpha / row)
        # theta(j) = theta(j) - sum of (alpha / m * (h(x) - y) * Xi)
        theta = np.subtract(theta, LaterItems)


    testdata = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(0, 385))
    testdata[:,0] = 1
    reference = np.dot(testdata, theta)
    testrow = testdata.shape[0]


    outputfile = open("output.csv", "wb")
    writer = csv.writer(outputfile)
    writer.writerow(['Id','reference'])
    resrow = 0

    for i in range(testrow):
        writer.writerow([resrow, reference[i][0]])
        resrow += 1
    
    outputfile.close()        
