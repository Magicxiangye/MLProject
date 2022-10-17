import Tree.trees
import trees
import matplotlib as plt
import treePlotter

if __name__ == '__main__':
    # 隐形眼镜数据集的决策树构建
    file_text = open('data/lenses.txt')

    # 构建向量
    lense = [inst.strip().split('\t') for inst in file_text.readlines()]

    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 构建决策树
    lense_tree = trees.createTree(lense, labels)
    print(lense_tree)
    # 画图
    treePlotter.createPlot(lense_tree)
