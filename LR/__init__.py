
import logsRre
import numpy as np

if __name__ == '__main__':
    dataArr, labelsMat = logsRre.loadDataSet()
    weight = logsRre.stocGradAscent0(dataArr, labelsMat)

    logsRre.plotBestFit(weight)