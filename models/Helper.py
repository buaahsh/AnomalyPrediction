class FeatureItem(object):

    """docstring for FeatureItem"""

    def __init__(self, line):
        super(FeatureItem, self).__init__()
        self.arg = line


def fitSklearnAll(X, y, model, multi=False):
    """
    Train a sklearn pipeline or core
    """
    model.fit(X, y)
    if multi:
        return {"pred": model.predict_proba(X), "index": 0}
    else:
        return {"pred": model.predict_proba(X)[:, 1], "index": 0}

def fitSklearnCV(X, y, cv, i, model, multi=False):
    """
    Train a sklearn pipeline or core -- wrapper to enable parallel CV.
    """
    import time

    tr = cv[i][0]
    vl = cv[i][1]
    model.fit(X.iloc[tr], y.iloc[tr])
    if multi:
        # start = time.clock()
        r = {"pred": model.predict_proba(X.iloc[vl]), "index": vl}
        return r
    else:
        start = time.clock()
        r = {"pred": model.predict_proba(X.iloc[vl])[:, 1], "index": vl}
        end = time.clock()
        print "The function run time is : %f seconds" %(end-start)
        print len(r["pred"])
        return r


def trainSklearn(model, grid, train, target, cv, refit=True, n_jobs=5, multi=False, evaluation=None):
    """
    Train a sklearn pipeline or core using textual data as input.
    """
    from joblib import Parallel, delayed
    from sklearn.grid_search import ParameterGrid
    from numpy import zeros
    if multi:
        from sklearn.metrics import accuracy_score
        # pred = zeros((train.shape[0],target.unique().shape[0]))
        pred = zeros((train.shape[0], target[0].unique().shape[0]))
        score_func = accuracy_score
    else:
        # from sklearn.metrics import roc_auc_score
        # score_func = roc_auc_score
        def evaluate(true, pred):
            from sklearn import metrics      
            fpr, tpr, thresholds = metrics.roc_curve(true, pred)
            return metrics.auc(fpr, tpr) 

        score_func = evaluate
        pred = zeros(train.shape[0])
    if evaluation:
        score_func = evaluation
    best_score = 0
    for g in ParameterGrid(grid):
        model.set_params(**g)
        if cv:
            if len([True for x in g.keys() if x.find('nthread') != -1]) > 0:
                results = [
                    fitSklearnCV(train, target, list(cv), i, model, multi) for i in range(cv.n_folds)]
            else:
                results = Parallel(n_jobs=n_jobs)(delayed(fitSklearnCV)(
                    train, target, list(cv), i, model, multi) for i in range(cv.n_folds))

            if multi:
                for i in results:
                    pred[i['index'], :] = i['pred']
                score = score_func(target, pred.argmax(1))
            else:
                for i in results:
                    pred[i['index']] = i['pred']
                score = score_func(target, pred)
        else:
            results = fitSklearnAll(train, target, model, multi)
            if multi:
                score = score_func(target, results['pred'].argmax(1))
            else:
                score = score_func(target, results['pred'])
        if score > best_score:
            best_score = score
            best_pred = pred.copy()
            best_grid = g


    print best_score
    print "Best Score: %0.5f" % best_score
    print "Best Grid", best_grid
    if refit:
        model.set_params(**best_grid)
        model.fit(train, target)
    return best_pred, model


# def loadTrainSet(dir='C:/Users/Shaohan/Documents/project/anomaly_prediction/data/RUBiSLogs/all/all.data'):
#rubis.txt-1.out
def loadTrainSet(dir='/Users/hsh/Documents/2015/AnomalyClassifier/y_out/i/all.data.out', filter=True):
    """
    Read in dataset to create training set.
    """
    import pandas as pd
    from pandas import DataFrame
    from sklearn.preprocessing import LabelEncoder
    X = pd.read_csv(dir)
    if filter:
        X = X[X.Label < 2]
    print "Length of instances:", X.shape[0]
    encoder = LabelEncoder()
    # y = DataFrame(encoder.fit_transform(X.iloc[:, -1]))
    y = X['Label']
    X = DataFrame(X.iloc[:, 1: -1])
    return X, y, encoder

# def loadRNNTrainSet(dir='C:/Users/Shaohan/Documents/project/anomaly_prediction/data/RUBiSLogs/all/all.data'):
def loadRNNTrainSet(dir='/Users/hsh/Downloads/all.data'):
    from numpy import array
    import pandas as pd
    from pandas import DataFrame
    from sklearn.preprocessing import LabelEncoder
    X = pd.read_csv(dir)
    X = DataFrame(X[X.Label < 2].iloc[:, 1: -1])
    X_max = DataFrame(X.describe()).loc['max', :]

    import numpy as np
    X = []
    y = []
    vocab_size = 13
    with open(dir, "r") as fIn:
        i = 0
        for line in fIn:
            line = line.strip()
            if line.endswith("2"):
                continue
            i += 1
            if i == 1:
                continue

            temp = np.zeros((vocab_size,1))
            tokens = line.split(",")
            for j, t in enumerate(tokens[1:-1]):
                if X_max[j] == 0:
                    temp[j] = 0
                else:
                    temp[j] = float(t) / X_max[j]
            X.append(temp)
            y.append(float(tokens[-1]))
    return X, y, None

def loadTestSet(dir='../data/test.json'):
    pass


if __name__ == "__main__":
    loadRNNTrainSet()