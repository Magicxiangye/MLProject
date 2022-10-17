from numpy import *
# GUI
from tkinter import *
import reTree

import matplotlib

# 像素图
matplotlib.use('TkAgg')
# plt的后端，用于刷新不同的树绘制图
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# 画图函数
def reDraw(tolS, tolN):
    reDraw.f.clf()  # 清空前端原来的图
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = reTree.createTree(reDraw.rawDat, reTree.modelLeaf, reTree.modelErr, (tolS, tolN))
        yHat = reTree.createForeCast(myTree, reDraw.testDat, reTree.modelTreeEval)
    else:
        myTree = reTree.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = reTree.createForeCast(myTree, reDraw.testDat)
    # 散点图
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)  # use scatter for data set
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # use plot for yHat
    reDraw.canvas.draw()


# 输入框值验证获取函数
def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


# 重画按钮的点击函数
def drawNewTree():
    tolN, tolS = getInputs()  # 获取参数
    reDraw(tolS, tolN)  # 画图


if __name__ == '__main__':
    root = Tk()

    reDraw.f = Figure(figsize=(5, 4), dpi=100)  # create canvas
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.draw()  # in mac use draw instead show
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

    Label(root, text="tolN").grid(row=1, column=0)
    tolNentry = Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    Label(root, text="tolS").grid(row=2, column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')
    Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
    chkBtnVar = IntVar()
    chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)

    reDraw.rawDat = mat(reTree.loadDataSet('data/sine.txt'))
    reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
    reDraw(1.0, 10)

    root.mainloop()
