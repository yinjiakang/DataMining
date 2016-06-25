import numpy as np
import itertools
import csv
# Ref: https://docs.python.org/3/library/csv.html
import sys
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from scipy.special import expit


if __name__ == '__main__':
    np.set_printoptions(edgeitems=5)

    mode = "TRAIN"

    if mode == "DEV":
        rows = 2000
        batchSize = 1000
        iteration = 1
        start = 0
    elif mode == "TEST":
        #rows = 1600750
        #rows = 576270
        rows = 57627 * 2
        batchSize = 6403
        iteration = 200
        start = 57627
    elif mode == "TRAIN":
        rows = 2177020
        batchSize = 6403
        iteration = 10
        start = 0

    cols = 11392 + 1
    """
    trainingFile = load_svmlight_file("train.txt", n_features=cols, dtype=np.int8)
    data = trainingFile[0]
    reference = trainingFile[1].reshape(rows, 1)
    print(reference.shape)
    """
    # Read in reference
    reference = np.genfromtxt("train.txt", dtype=np.int8, usecols=(0), max_rows=rows)
    y = reference.reshape(rows, 1)
    #theta = np.zeros((cols, 1))
    theta = np.genfromtxt("theta.csv", delimiter=' ').reshape(cols, 1)
    alpha = 0.5

    # Read in data
    
    for curRow in range(start, rows, batchSize):
        print("At batch %d\n" %(curRow))
        with open("train.txt") as trainingFile:
            # Ref: http://stackoverflow.com/questions/19031423/how-to-loop-through-specific-range-of-rows-with-python-csv-reader
            reader = csv.reader(itertools.islice(trainingFile, curRow, curRow + batchSize), delimiter=' ')
            rowNumber = 0
            data = []
            colIdx = []
            rowIdx = []

            for row in reader:
                elementList  = [ele.split(':') for ele in row[1:]]
                subData = [1] + [int(ele[1]) for ele in elementList]
                subColIdx = [0] + [int(ele[0]) for ele in elementList]
                subRowIdx = [rowNumber] * (len(elementList) + 1)

                rowNumber += 1
                data.extend(subData)
                colIdx.extend(subColIdx)
                rowIdx.extend(subRowIdx)

            # Ref: http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
            # Add one col as x0
            partialX = csr_matrix((data, (rowIdx, colIdx)), shape=(batchSize, cols))
            partialY = y[curRow:curRow + batchSize]
        
        for it in range(iteration):
            # Sigmoid Ref: http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
            h = expit(partialX.dot(theta))
            J = np.sum(np.multiply(partialY, np.log(h)) + np.multiply(1 - partialY, np.log(1 - h))) / -batchSize 
            print("At iteration %d: J = %f\n" %(it, J))
            partialDerivativeJ = np.sum(partialX.multiply(h - partialY), axis=0).reshape(cols, 1)
            theta = theta - alpha / rows * partialDerivativeJ

    # Save theta
    with open('theta.csv', 'w') as file:
        for i in theta:
            file.write("%f\n" %(i[0]))

    # Predict
    if mode == "DEV":
        testFile = load_svmlight_file("stest.txt", n_features=cols, dtype=np.int8)
        realTestFile = load_svmlight_file("stest.txt", n_features=cols, dtype=np.int8)
    elif mode == "TEST":
        # Validation
        testFile = load_svmlight_file("testTest.txt", n_features=cols, dtype=np.int8)
        realTestFile = load_svmlight_file("test.txt", n_features=cols, dtype=np.int8)
    elif mode == "TRAIN":
        testFile = load_svmlight_file("test.txt", n_features=cols, dtype=np.int8)

    testData = testFile[0]
    testRows = testData.shape[0]
    testRef = testFile[1].reshape(testRows, 1)
    testH = expit(testData.dot(theta))
    testY = np.greater(testH, 0.5)

    realTestData = realTestFile[0]
    realTestRows = realTestData.shape[0]
    realTestRef = realTestFile[1].reshape(realTestRows, 1)
    realTestH = expit(realTestData.dot(theta))
    realTestY = np.greater(realTestH, 0.5)

    correctCount = np.count_nonzero(testRef == testY)

    # Print result
    with open('testResult.csv', 'w') as file:
        file.write("Correct Rate: %f\n" %(correctCount / testRows))
        file.write("id,label\n")
        for i in range(0, testRows):
            file.write("%d %d\n" %(i, testY[i]))

    with open('result.csv', 'w') as file:
        file.write("id,label\n")
        for i in range(0, realTestRows):
            file.write("%d,%d\n" %(i, realTestY[i]))