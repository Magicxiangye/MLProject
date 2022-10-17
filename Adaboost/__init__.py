

import numpy as np
import matplotlib .pyplot as plt
import adaboost

if __name__ == '__main__':
    datArr, labels = adaboost.loadDataSet('data/horseColicTraining2.txt')

    cla, aggl = adaboost.adaBoostTrainDS(datArr, labels, 10)

    adaboost.plotROC(aggl.T, labels)