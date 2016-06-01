import sys
from numpy import *
import numpy as np
from math import *


def sigmoid(m):
    return 1 / (1 + np.exp(-m))

def func(k):
    return k * (1 - k)

if __name__ == "__main__":

    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    #get reference ,that is , y(i)
    refer = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385))

    row = data.shape[0]
    col = data.shape[1]
    refer = refer.reshape(row, 1)
    y = refer

    maxNumber = y.max()
    minNumber = y.min()

    y = (y - minNumber) / (maxNumber - minNumber)


    
    #nodes = ceil(sqrt(col))
    nodes = 160


    """
    hiddenWeight = np.random.randn(col, nodes) * 0.01
    outputWeight = np.random.randn(nodes, 1) * 0.01
    hiddenTheta = np.zeros((1, nodes),  dtype=np.float)
    outputTheta = 0
    

    """

    hiddenWeight = np.genfromtxt("hiddenWeight.csv", delimiter=',')
    outputWeight = np.genfromtxt("outputWeight.csv", delimiter=',')    
    hiddenTheta = np.genfromtxt("hiddenTheta.csv", delimiter=',', usecols = range(0, nodes))
    outputT = np.genfromtxt("outputTheta.csv", delimiter=',', usecols = (0))
    #getL = np.genfromtxt("l.csv", delimiter=',', usecols = (0))
    
    outputWeight = outputWeight.reshape(nodes, 1)
    hiddenWeight = hiddenWeight.reshape(col, nodes)
    hiddenTheta = hiddenTheta.reshape(1, nodes)
    outputTheta = array([outputT])
    outputTheta = outputTheta.reshape(1,1)
    #l = array([getL])
    #l = l.reshape(1,1)
    

    """
    l = 0.4
    iteration = 3000
    lastJ = sys.maxsize

    for i in range(0, iteration):
        for r in range(row):
            thisRow = data[r]

            # 1 * nodes
            hiddenI = np.dot(thisRow, hiddenWeight) + hiddenTheta
            # 1 * nodes
            hiddenX = sigmoid(hiddenI)


            # 1 * nodes
            outputI = np.dot(hiddenX, outputWeight) + outputTheta
            # 1 * nodes
            outputX = sigmoid(outputI)


            outputErr = func(outputX) * (y[r] - outputX)
            # nodes * 1
            hiddenErr = func(hiddenX) * outputErr * np.transpose(outputWeight)


            outputTheta += l * outputErr
            # nodes * 1
            outputWeight += l * outputErr * hiddenX.reshape(nodes, 1)

            hiddenTheta += l * hiddenErr
            # 1 * col + nodes * 1  -> col * nodes
            hiddenWeight += l * np.outer(data[r], hiddenErr)

        hidden = sigmoid(np.dot(data, hiddenWeight) + hiddenTheta)
        output = sigmoid(np.dot(hidden, outputWeight) + outputTheta)
        output = output * (maxNumber - minNumber) + minNumber
        J = np.sum(np.square(output - refer)) / (row * 2)

        if J > lastJ:
            l = l / 2
            if l <= 0.02:
                break
        lastJ = J
        print (i, J)


    """


    testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    #testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 6), max_rows=7)
    testM = testData.shape[0]

    testHidden = sigmoid(np.dot(testData, hiddenWeight) + hiddenTheta)
    testOutput = sigmoid(np.dot(testHidden, outputWeight) + outputTheta)
    testOutput = testOutput * (maxNumber - minNumber) + minNumber

    with open('ans.csv', 'w') as file:
        file.write("id,reference\n")
        for i in range(0, testM):
            file.write("%d,%f\n" %(i, testOutput[i]))


    """

    with open('hiddenWeight.csv', 'w') as File:
        for i in range (0, col):
            for j in range(0, nodes - 1):
                File.write("%.10f," %hiddenWeight[i][j])
            File.write("%.10f\n" %hiddenWeight[i][nodes - 1])
        
    with open('outputWeight.csv', 'w') as File:
        for i in range(0, nodes - 1):
            File.write("%.10f," %outputWeight[i][0])
        File.write("%.10f\n" %outputWeight[nodes - 1][0])

    with open('hiddenTheta.csv', 'w') as File:
        for i in range(0, nodes):
            File.write("%.10f," %hiddenTheta[0][i])
     
    with open('outputTheta.csv', 'w') as File:
        File.write("%.10f," %outputTheta)

    with open('l.csv', 'w') as File:
        File.write("%.10f," %l)
    """
    