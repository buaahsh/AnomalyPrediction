from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline

from Helper import *
from models.Unitils import *


# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    # ('feat', SelectPercentile(chi2)),
    # ('core', SVC(probability=True))
    # ('core', tree.DecisionTreeClassifier())
    # ('core', GaussianNB())
    # ('core', LinearSVC())
    ('core', KNeighborsClassifier(n_neighbors=15))
    # ('core', GradientBoostingClassifier(n_estimators=20, min_samples_split=10))
    # ('core', AdaBoostClassifier(n_estimators=20))
])

grid = {
    # 'tfidf__ngram_range':[(2,6)],
    # 'feat__percentile':[95,90,85],
    # 'model__C':[0.5,1,2,3,4,5,7.5,10]
    # 'model__n_neighbors' : [i for i in xrange(20, 40)] 
    # 'model__alpha' : [0, 0.05, 0.1] 
}

def evaluate(true, pred):
    from sklearn import metrics
    import matplotlib.pyplot as plt

    DumpObj(true, "15RNN.true")
    DumpObj(pred, "15RNN.pred")

    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    plt.plot(fpr, tpr)
    plt.show()
    return metrics.auc(fpr, tpr) 

def main():
    # load data
    train, target, encoder = loadTrainSet()
    cv = KFold(train.shape[0], n_folds=4, shuffle=True)
    # cv = None
    # pred, core = trainSklearn(pipe,grid,train,target,cv,n_jobs=2,multi=True)

    pred, model = trainSklearn(pipe,grid,train,target,cv,n_jobs=2,multi=False, evaluation=evaluate)

    # pred, core = trainSklearn(pipe,grid,train,target,cv,n_jobs=2,multi=False)

    # core = SVC(probability=True).fit(train,target)
    # z = {"pred": core.predict_proba(train), "index":0}
    # from numpy import zeros
    # from sklearn.metrics import accuracy_score
    # pred = zeros((train.shape[0], target[0].unique().shape[0]))

    # pred = z['pred']
    # score = accuracy_score(target,pred.argmax(1))
    # print score
    

if __name__ == "__main__":
    main()
