import cPickle as p  


def DumpObj(obj, fileName):
    f = file(fileName, 'w')  
    p.dump(obj, f) # dump the object to a file  
    f.close()


def LoadObj(fileName):
    f = file(fileName)
    obj = p.load(f)
    return obj


def ComputeLeadtime(true, pred):
    res = []
    isAlert = 0
    temp = 0
    for t, p in zip(true, pred):
        if t == 1:
            isAlert += 1
            if p > 0.2 and temp == 0:
                temp = isAlert
        else:
            if temp != 0:
                res.append(min(isAlert - temp, 50))
            isAlert = 0
            temp = 0
    if temp:
        res.append(min(isAlert - temp, 50))
    return (sum(res) + len(res)) * 1.0 / len(res)


def MultiPlot():
    import matplotlib.pyplot as pl
    def SinglePlot(ax, arg, label):
        from sklearn import metrics      
        true = LoadObj(label + ".true")
        pred = LoadObj(label + ".pred")
        fpr, tpr, thresholds = metrics.roc_curve(true, pred)
        print label, metrics.auc(fpr, tpr)
        # print sum(true[0])
        print label, ComputeLeadtime(true[0], pred)
        ax.plot(fpr, tpr, arg, label=label)
    ax = pl.subplot()
    SinglePlot(ax, "y", "RNN")
    SinglePlot(ax, "r", "KNeighbors")
    SinglePlot(ax, "b", "GaussianNB")
    SinglePlot(ax, "g", "GradientBoosting")
    

    ax.plot([0, 0.5, 1], [0, 0.5, 1], "k--", label="Random Chance")

    ax.legend(loc=4)
    pl.xlabel('False Positive Rate')# make axis labels
    pl.ylabel('True Positive Rate')
    # pl.xlim(0, 1)# set axis limits
    # pl.ylim(0, 1)
    pl.show()# show the plot on the screen

if __name__ == "__main__":
    MultiPlot()
    # true = [0,0,1,1,1,1,0,0,0,0,1,1,1,0]
    # pred = [0,0,0,1,1,1,0,0,0,0,1,0,1,0]
    # print ComputeLeadtime(true, pred)