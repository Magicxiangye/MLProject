import operator
import kNN

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    datingDataMat, datingLabel = kNN.file2matrix('data/datingTestSet2.txt')

    fig = plt.figure()
    # 就一个图
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15 * np.array(datingLabel), 15 * np.array(datingLabel))
    plt.show()

