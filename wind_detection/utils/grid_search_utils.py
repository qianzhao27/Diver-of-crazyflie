import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score

def report_model_score(clf, X, y, cv=3):
    scores = cross_val_score(clf, X, y, cv=3, n_jobs=-1)
    print('Model Report: ')
    print('Mean cv score: {:.3f} +/- std: {:.3f}'.format(np.mean(scores), np.std(scores)))
    
def model_selection(clf, param_grid, X , y):
    cv = KFold(n_splits=3, shuffle=True, random_state=13)
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(X, y)
    return grid_search

def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f}, Standard Deviation: {1:.4f}".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def evaluate_param(clf, parameter, param_values, index, X_train, y_train):
    grid_search = GridSearchCV(clf, param_grid={parameter: param_values}, cv=3)
    grid_search.fit(X_train, y_train)
    
    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]
       
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
 
    plt.subplot(4,2, index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    plt.grid(True)
    
    plt.show()