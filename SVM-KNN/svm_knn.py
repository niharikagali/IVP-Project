import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from multiprocessing import Pool
from matplotlib.colors import to_hex
from matplotlib import rc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MultiLabelBinarizer
from os.path import basename
from collections import defaultdict
from multiprocessing import Pool
import svmlight
from sklearn.neighbors import KNeighborsClassifier
import re
from multiprocessing import Pool
from collections import defaultdict
from sklearn.metrics import f1_score

pd.set_option('precision', 3) 

sns.set_palette("husl")
COLORS = sns.color_palette()
COLOR_REU = COLORS[0]
COLOR_OHS = COLORS[3] 
rc('text', usetex=True)


def read_files_in_cat(dir):
    for fp in glob.glob(dir + '/*'):
        with open(fp, 'r') as file:
            yield (basename(fp), file.read())

            
def read_set(dir_with_categories):
    for cat_dir in glob.glob(dir_with_categories + '/*'):
        cat_name = basename(cat_dir)
        if cat_name != 'unknown':
            for doc_name, doc_text in read_files_in_cat(cat_dir):
                yield (cat_name, doc_name, doc_text)

def read_doc_files(dir_with_categories):
    labels = defaultdict(list)
    texts = {}
    for cat_name, doc_name, doc_text in read_set(dir_with_categories):
        if doc_name not in texts:
            texts[doc_name] = doc_text
        labels[doc_name].append(cat_name)

    for doc_name in texts.iterkeys():
        yield (texts[doc_name], labels[doc_name], doc_name)


def best_ks(tup):
    X_train, y_train, X_test, y_test, label, k = tup

    if k:
        kb = SelectKBest(mutual_info_classif, k=k)
        X_train_best = kb.fit_transform(X_train, y_train)
        X_test_best  = kb.transform(X_test)
        return k, label, X_train_best, y_train, X_test_best, y_test
    else:
        return k, label, X_train, y_train, X_test, y_test

def read_dataset(path, best_features=[500,1000,2000,5000,10000,None]):
    test_texts, test_labels, test_doc_names    = zip(*read_doc_files(path + '/test'))
    train_texts, train_labels, train_doc_names = zip(*read_doc_files(path + '/training'))
    

    vectorizer = TfidfVectorizer(min_df=3, norm='l2', use_idf=True, stop_words='english', decode_error='ignore')
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(train_labels)
    Y_test  = mlb.transform(test_labels)
    
    args = [(X_train, Y_train[:,label_col], X_test, Y_test[:,label_col], mlb.classes_[label_col],k)
            for k in best_features
            for label_col in range(len(mlb.classes_))]
    
    pool = Pool(8)
    res = defaultdict(dict)
    for k, label, X_train, y_train, X_test, y_test in pool.map(best_ks, args):
        res[k][label] = (X_train, y_train, X_test, y_test)
        
    pool.terminate()
    
    return res


def to_svmlight_format(sparse_mat, labels):
    lil = sparse_mat.tolil()
    return [(label, zip(row, data))
            for (row, data, label) in zip(lil.rows, lil.data, labels)]

def svm(train_docs, train_labels, test_docs, params):
    
    kernel, param = params
    train_docs_svm = to_svmlight_format(train_docs, (1 if l == 1 else -1 for l in train_labels))
    test_docs_svm  = to_svmlight_format(test_docs, np.zeros(test_docs.shape[0]))
    
    if kernel == 'rbf':
        model = svmlight.learn(train_docs_svm, type='classification',
                               kernel='rbf', rbf_gamma=param)
    elif kernel == 'poly' or kernel == 'polynomial':
        model = svmlight.learn(train_docs_svm, type='classification',
                               kernel='polynomial', poly_degree=param)
    else:
        raise ValueError('Unsupported svm parameters: ' + str(params))
  
    margins = svmlight.classify(model, test_docs_svm)
    predict_labels = [1 if p > 0 else 0 for p in margins]
    
    return predict_labels


def kNN(train_docs, train_labels, test_docs, params):
    k = params
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_docs, train_labels)
    predict_labels = neigh.predict(test_docs)
    return predict_labels


KNN_VERS     = [('k-NN', k) for k in [1,15,30,45,60]]
SVM_VERS     = [('SVM', params)
                for params in [('poly',deg)   for deg   in [1,2,3,4,5]] +
                              [('rbf', gamma) for gamma in [0.6,0.8,1.0,1.2]]]

BEST_FEATURES = [500, 1000, 2000, 5000, 10000, None]
ALGO_FNS = {'k-NN': kNN,'SVM': svm}


class Algo():
    
    def __init__(self, name, params, features, dataset_name):
        self.name = name
        self.dataset_name = dataset_name
        self.features = features
        self.params = params
        
    def __repr__(self):
        if self.params != None:
            if isinstance(self.params,list) or isinstance(self.params,tuple):
                params_name = '_'.join(map(str, self.params))
            else:
                params_name = str(self.params)
        else:
            params_name = ''
        
        return self.name + ' ' + params_name + (' ' if self.params != None else '') +               str(self.features) + 'F ' + self.dataset_name

    @staticmethod
    def from_filepath(fp):
        base = re.sub(' (predict|true)_labels.csv','', os.path.basename(fp))
        
        parts = base.split(' ')
        if len(parts) == 3:
            algo_name, features, dataset_name = parts
            params = None
        elif len(parts) == 4:
            algo_name, params, features, dataset_name = parts
        else:
            raise ValueError('Unexpected filepath: ' + fp)

        return Algo(algo_name, params,
                    None if features == 'NoneF' else int(re.sub('F','',features)),
                    dataset_name)

def predict_algo(algo):
    print(algo)
    
    label_data = DATASETS[algo.dataset_name][algo.features]
    labels = list(label_data.iterkeys())
    
    predicts = defaultdict(dict)
    for label, (X_train, y_train, X_test, y_test) in label_data.iteritems():
        predicts[label] = ALGO_FNS[algo.name](X_train, y_train, X_test, algo.params)
    
    pd.DataFrame(
        data  = [label_data[l][3] for l in labels], 
        index = labels
    ).to_csv('results/' + str(algo) + ' true_labels.csv')
    
    pd.DataFrame(
        data  = map(predicts.get, labels),
        index = labels
    ).to_csv('results/' + str(algo) + ' predict_labels.csv')
    
    print()
    return str(algo) + ' DONE'

def missing(algos_to_run):
    already_run = set(map(lambda fp: str(Algo.from_filepath(fp)), glob.glob('results/*')))
    return list(filter(lambda algo: str(algo) not in already_run, algos_to_run))


algos_to_run = [Algo(name, params, best_feat, 'r')
                for best_feat in BEST_FEATURES
                for name, params in (SVM_VERS + KNN_VERS)]

USE_MULTIPLE_CORES = True

if USE_MULTIPLE_CORES:
    pool = Pool(processes=8)
    result = pool.map(predict_algo, missing(algos_to_run))
    pool.terminate()
else:
    map(predict_algo, missing(algos_to_run))


def f1s_from_preds(dataset_name):
    
    algo_series = []
    f1_micro = {}
    
    for fp in filter(lambda fp: dataset_name in fp, glob.glob('results/*')):
        
        algo = Algo.from_filepath(fp)
        
        if str(algo) not in f1_micro:
            
            predicts = pd.read_csv('results/' + str(algo) + ' predict_labels.csv', index_col=0)
            trues    = pd.read_csv('results/' + str(algo) + ' true_labels.csv', index_col=0)
            algo_series.append(
                pd.Series([f1_score(trues.loc[label], predicts.loc[label])
                           for label in predicts.index],
                          index=predicts.index, name=str(algo))
            )

            f1_micro[str(algo)] = f1_score(trues, predicts, average='micro')
    
    df = pd.concat(algo_series, axis=1)
    return df.sort_index(axis=1), f1_micro


def best(df, f1_micro, list_of_keywords):   
    cols = []
    for keywords in list_of_keywords:
        ks = keywords if isinstance(keywords, tuple) or isinstance(keywords,list) else [keywords]
        filtered = dict(
            filter(lambda algo_f1: all(str(k) in algo_f1[0] for k in ks),
                   f1_micro.iteritems())
        )
        best_f1 = max(filtered, key=filtered.get)
        cols.append(best_f1)
        
    return df[cols], {c: f1_micro[c] for c in cols}


def sum_up(df, f1_micro):
    svms = [['SVM','poly_' + str(d), 'r'] for d in range(1,6)] +           [['SVM','rbf_' + str(g), 'r']  for g in [0.6,0.8,1.0,1.2]]
    cats = [ ['k-NN','r']] + svms
    
    best_df, best_f1_micro = best(df, f1_micro, cats)
    
    return best_df.loc[p].append(pd.Series(best_f1_micro, name='F1 microavg.')).round(2)
    

def final(df, f1_micro, categories=None, dropzeros=False):
    svms = [['SVM','poly_' + str(d)] for d in range(1,6)] +           [['SVM','rbf_' + str(g)] for g in [0.6,0.8,1.0,1.2]]
    algos = ['k-NN'] + svms
    
    best_df, best_f1_micro = best(df, f1_micro, algos)
    
    if categories:
        best_df = best_df.loc[categories]
        
    return best_df, pd.Series(best_f1_micro, name='F1 microavg.')
    

DF_REU, F1_REU = f1s_from_preds('r');
DF_OHS, F1_OHS = f1s_from_preds('o');


def pval_matrix(df, f1_micro, test, alpha):
    
    res = defaultdict(list)
    
    for algo1 in df:
        for algo2 in df:
            res[algo1].append(test(df[algo1], df[algo2]).pvalue)
    
    clean_names = [algo.name + (' ' + str(algo.params) if algo.params else '')
                   for algo in map(Algo.from_filepath, df.columns)]
    
    df_pvals = pd.DataFrame.from_dict(res)[df.columns].round(3)
    
    clean_to_full = dict(zip(clean_names, df_pvals.columns))
    
    def color_cell(f1_row, f1_col):
        if f1_row < f1_col:
            return 'background-color: '  + to_hex(COLORS[4]) + '22' # blue
        else:
             return 'background-color: ' + to_hex(COLORS[5]) + '22' # red
    
    def color_col(col):
        stylings = ['' if pval > alpha
                    else color_cell(f1_micro[clean_to_full[row_name]], f1_micro[clean_to_full[col.name]])
                    for (pval, row_name) in zip(df_pvals[col.name], df_pvals.columns)]
        return stylings

    df_pvals.index = clean_names
    df_pvals.columns = clean_names
    df_pvals_styled = df_pvals.style.apply(color_col, axis=0)

    return df_pvals_styled

