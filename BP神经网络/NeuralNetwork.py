import sys
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

    nodes = ceil(sqrt(col))

    hiddenWeight = np.random.rand(col, nodes)
    outputWeight = np.random.rand(nodes, 1)
    hiddenTheta = np.random.rand(1, nodes)
    outputTheta = np.random.rand()

    l = 0.4
    iteration = 10
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
            break
        lastJ = J
        print i, J

    testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    #testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 6), max_rows=7)
    testM = testData.shape[0]

    testHidden = sigmoid(np.dot(testData, weight0) + hiddenTheta)
    testOutput = sigmoid(np.dot(testHidden, weight1) + outputTheta)
    testOutput = testOutput * (maxNumber - minNumber) + minNumber

    with open('ans', 'w') as file:
        file.write("id,reference\n")
        for i in range(0, testM):
            file.write("%d,%f\n" %(i, testOutput[i]))
     
