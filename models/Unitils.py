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
            if p > 0.5 and temp == 0:
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
        true = LoadObj("t/" + label + ".true")
        pred = LoadObj("t/" + label + ".pred")
        fpr, tpr, thresholds = metrics.roc_curve(true, pred)
        # print label, metrics.auc(fpr, tpr)
        # print sum(true[0])
        # print label, ComputeLeadtime(true[0], pred)
        if arg == 'k-':
            ax.plot(fpr, tpr, arg, label=label, dashes=[8, 4, 2, 4, 2, 4], linewidth=2)
        else:
            ax.plot(fpr, tpr, arg, label=label, linewidth=2)
        # with open('ibm/' + label, 'w') as f_out:
        #     for f,t in zip(fpr, tpr):
        #         print >>f_out, '{0}\t{1}'.format(f, t)
    ax = pl.subplot()
    SinglePlot(ax, "k--", "AvgCycRNN")
    SinglePlot(ax, "k-.", "SeqCycRNN")
    #
    SinglePlot(ax, "k", "SeqRNN")
    SinglePlot(ax, "k-", "AvgRNN")
    # SinglePlot(ax, "k--", "KNeighbors")
    # SinglePlot(ax, "k-.", "Boosting")
    # SinglePlot(ax, "k:", "GaussianNB")
    # # SinglePlot(ax, "k", "RNN-3")
    # SinglePlot(ax, "y", "RNN-5")
    # SinglePlot(ax, "r", "RNN-10")
    # SinglePlot(ax, "g", "RNN-15")
    #
    #
    ax.plot([0, 0.5, 1], [0, 0.5, 1], "k", label="Random")
    #
    ax.legend(loc=4)
    pl.xlabel('False Positive Rate')# make axis labels
    pl.ylabel('True Positive Rate')
    pl.xlim(0, 1)# set axis limits
    pl.ylim(0, 1)
    pl.show()# show the plot on the screen



def ef():
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + 0.1, height + 0.2, '%s' % float(height))

    import matplotlib.pyplot as plt
    import numpy as np
    # quants: GDP
    # labels: country name
    labels   = []
    quants   = []
    # Read data
    labels = ["GaussianNB", "Boosting", "KNeighbors", "AvgRNN", "SeqRNN","AvgCycRNN" ,"SeqCycRNN"]
    quants = [1.13,3.67,16.5,4.46,4.32,5.45,5.12]

    width = 0.6
    ind = np.linspace(0.5,9.5,7)
    # make a square figure
    fig = plt.figure(1, figsize=(12,6))
    ax  = fig.add_subplot(111)
    # Bar Plot
    b = ax.bar(ind-width/2,quants,width,color='coral')

    # Set the ticks on x-axis
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    # labels
    ax.set_ylabel('Execution time(ms)')
    # ax.set_ylabel('GDP (Billion US dollar)')
    # title
    # ax.set_title('Top 10 GDP Countries', bbox={'facecolor':'0.8', 'pad':5})
    autolabel(b)
    plt.show()

def maxleadtime():
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + 0.1, height + 0.2, '%s' % float(height))

    import matplotlib.pyplot as plt
    import numpy as np

    ax = plt.subplot()
    y = [0.5332, 0.6032, 0.6523, 0.7420, 0.7594, 0.6851, 0.7421, 0.6821, 0.57]
    x = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    l=[20,100,0,1]
    ax.axis(l)

    ax.plot(x, y, '-or', label="GaussianNB")

    y = [0.52, 0.6632, 0.63, 0.7420, 0.7794, 0.6451, 0.6921, 0.7821, 0.67]
    ax.plot(x, y, '-xy', label='KNeighbors')

    y = [0.721, 0.7232, 0.8323, 0.8320, 0.8747, 0.8432, 0.8621, 0.8421, 0.83]
    ax.plot(x, y, '-^c', label='Boosting')

    y = [0.54, 0.6932, 0.7833, 0.8320, 0.8734, 0.8832, 0.8521, 0.7921, 0.73]
    ax.plot(x, y, '-*b', label="SeqRNN")

    y = [0.632, 0.8032, 0.8523, 0.890, 0.9581, 0.9432, 0.9521, 0.9221, 0.94]
    ax.plot(x, y, '-+g', label='SeqCycRNN')

    ax.legend(loc=4)
    plt.xlabel('Max Lead Time')# make axis labels
    plt.ylabel('AUC')

    # labels
    # title
    # ax.set_title('Top 10 GDP Countries', bbox={'facecolor':'0.8', 'pad':5})
    plt.show()

if __name__ == "__main__":
    # maxleadtime()
    # ef()
    MultiPlot()
    # true = [0,0,1,1,1,1,0,0,0,0,1,1,1,0]
    # pred = [0,0,0,1,1,1,0,0,0,0,1,0,1,0]
    # print ComputeLeadtime(true, pred)