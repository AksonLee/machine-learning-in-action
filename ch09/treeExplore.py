import numpy as np
import tkinter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import regTrees

def tkinterTestMy():
    root = tkinter.Tk()
    myLabel = tkinter.Label(root, text = 'Hello World!')
    myLabel.grid()
    root.mainloop()

def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = regTrees.createTree(reDraw.rowData, regTrees.modelLeaf, regTrees.modelErr, ops = (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testData, regTrees.modelTreeEval)

    else:
        myTree = regTrees.createTree(reDraw.rawData, ops = (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testData)

    reDraw.a.scatter(np.array(reDraw.rawData[:, 0]), np.array(reDraw.rawData[:, 1]), s = 5)
    reDraw.a.plot(reDraw.testData, yHat, linewidth = 2.0)
    reDraw.canvas.draw()

def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print('enter integer for tolN')
        tolNentry.delete(0, tkinter.END)
        tolNentry.insert(0, '10')
    
    try:
        tolS = int(tolSentry.get())
    except:
        tolS = 10
        print('enter integer for tolS')
        tolSentry.delete(0, tkinter.END)
        tolSentry.insert(0, '1.0')
    
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

root = tkinter.Tk()

reDraw.f = Figure(figsize=(5,4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

tkinter.Label(root, text = 'Plot Place Holder').grid(row = 0, columnspan = 3)
tkinter.Label(root, text = 'tolN').grid(row = 1, column = 0)
tolNentry = tkinter.Entry(root)
tolNentry.grid(row = 1, column = 1)
tolNentry.insert(0, '10')

tkinter.Label(root, text = 'tolS').grid(row = 2, column = 0)
tolSentry = tkinter.Entry(root)
tolSentry.grid(row = 2, column = 1)
tolSentry.insert(0, '1.0')

tkinter.Button(root, text = 'ReDraw', command = drawNewTree).grid(row = 1, column = 2, rowspan = 3)

chkBtnVar = tkinter.IntVar()
chkBtn = tkinter.Checkbutton(root, text = 'model tree', variable = chkBtnVar)
chkBtn.grid(row = 3, column = 0, columnspan = 2)

reDraw.rawData = np.mat(regTrees.loadDataSet('sine.txt'))
reDraw.testData = np.arange(np.min(reDraw.rawData[:, 0]), np.max(reDraw.rawData[:, 0]), 0.01)

reDraw(1.0, 10)

tkinter.Button(root, text = 'Quit', fg = 'black', command = root.quit).grid(row = 1, column = 2)

root.mainloop()
