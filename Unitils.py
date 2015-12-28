import cPickle as p  


def DumpObj(obj, fileName):
    f = file(fileName, 'w')  
    p.dump(obj, f) # dump the object to a file  
    f.close()


def LoadObj(fileName):
    f = file(fileName)
    obj = p.load(f)
    return obj

def MultiPlot():
    import matplotlib.pyplot as pl
    def SinglePlot(ax, arg, label):
        from sklearn import metrics      
        true = LoadObj(label + ".true")
        pred = LoadObj(label + ".pred")
        fpr, tpr, thresholds = metrics.roc_curve(true, pred)
        ax.plot(fpr, tpr, arg, label=label)
    ax = pl.subplot()
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