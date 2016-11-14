"""Blend features extracted with Conv Nets and make predictions/submissions."""
from __future__ import division, print_function
from datetime import datetime
from glob import glob
import os
import sys

PY2 = sys.version_info[0] == 2
if PY2:
    import cPickle as pickle
else:
    import pickle as pickle

import click
import numpy as np
import pandas as pd
import theano
from lasagne import init
from lasagne.updates import adam
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import DenseLayer, InputLayer, FeaturePoolLayer, LocalResponseNormalization2DLayer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nolearn.lasagne import BatchIterator
from nolearn.lasagne.handlers import SaveWeights
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import yaml
from sklearn.externals import joblib
from skll import metrics
from scipy import stats

import sys
sys.path.insert(0, 'src/utils/')
sys.path.insert(0, 'src/configs/')

import data
import nn
import util

np.random.seed(9)

START_LR = 0.005
END_LR = START_LR #* 0.001
L1 = 2e-5
L2 = 0.005
N_ITER = 100
PATIENCE = 20
POWER = 0.5
N_HIDDEN_1 = 32
N_HIDDEN_2 = 32
BATCH_SIZE = 12

SCHEDULE = {
    60: START_LR ,#/ 10.0,
    80: START_LR ,#/ 100.0,
    90: START_LR ,#/ 1000.0,
    N_ITER: 'stop'
}


class BlendNet(nn.Net):

    def set_split(self, files, labels):
        """Override train/test split method to use our default split."""
        def split(X, y, eval_size):
            if eval_size:
                tr, te = data.split_indices(files, labels, eval_size)
                return X[tr], X[te], y[tr], y[te]
            else:
                return X, X[len(X):], y, y[len(y):]
        setattr(self, 'train_test_split', split)  # train_split?


class ResampleIterator(BatchIterator):

    def __init__(self, batch_size, resample_prob=0.2, shuffle_prob=0.5):
        self.resample_prob = resample_prob
        self.shuffle_prob = shuffle_prob
        super(ResampleIterator, self).__init__(batch_size)

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        indices = data.balance_per_class_indices(self.y.ravel())
        for i in range((n_samples + bs - 1) // bs):
            r = np.random.rand()
            if r < self.resample_prob:
                sl = indices[np.random.randint(0, n_samples, size=bs)]
            elif r < self.shuffle_prob:
                sl = np.random.randint(0, n_samples, size=bs)
            else:
                sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

def estimator(protocol, classifier, n_features, files, X, labels, run, fold, eval_size=0.1):

    final_weights = 'weights/final_%s_%s_fold_%s.pkl' % (classifier, run, fold)

    if classifier == "SVM":
        if os.path.exists(final_weights):
            est = joblib.load(final_weights)

        else:
            svm = SVC(kernel='linear', class_weight='balanced', cache_size=5500, probability=True)
            if protocol != 'protocol3':
                svm_model = svm
                param_grid = {"C": [1e-3,1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]}
                cv = StratifiedShuffleSplit(labels.reshape((labels.shape[0],)), n_iter=10, test_size=0.1, random_state=0)
                est = GridSearchCV(svm_model, param_grid=param_grid, scoring='roc_auc', n_jobs=15, cv=cv, verbose=2)
                est.fit(X, labels.reshape((labels.shape[0],)))
            else:
                param_grid = {"estimator__C": [1e-3,1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]}
                binarized_labels = label_binarize(np.squeeze(labels), classes=[0,1,2])
                svm_model = OneVsRestClassifier(svm)
                cv = StratifiedShuffleSplit(binarized_labels, n_iter=10, test_size=0.1, random_state=0)
                est = GridSearchCV(svm_model, param_grid=param_grid, scoring='roc_auc', n_jobs=15, cv=cv, verbose=2)
                est.fit(X, binarized_labels)

            est = est.best_estimator_
            print("Best estimator found by grid search for %s: " % (classifier))
            print(est)

            # Persistence
            #joblib.dump(est, final_weights)

    elif classifier == "RF":
        if os.path.exists(final_weights):
            est = joblib.load(final_weights)

        else:
            #for criterion in ["gini","entropy"]:
            #                    for n_estimators in [10, 50, 100, 200]:#, 200, 250, 500, 750, 1000]:
            #                            for max_features in [None]: #"auto", "sqrt", "log2",
            #                                    # We are not using class_weight='auto'. Error in sklearn

            param_grid = {'criterion': ['gini','entropy'],
                          'n_estimators': [50, 100, 200, 300, 10, 250, 500, 750]}
            est = GridSearchCV(RandomForestClassifier(max_features="auto"),
                               param_grid=param_grid, n_jobs=-1, verbose=2)
            print(X[:3])
            est.fit(X, labels.reshape((labels.shape[0],)))

            est = est.best_estimator_
            print("Best estimator found by grid search for %s: " % (classifier))
            print(est)

            # Persistence
            joblib.dump(est, final_weights)

    else:
        layers = [
            (InputLayer, {'shape': (None, n_features)}),
            (DenseLayer, {'num_units': N_HIDDEN_1, 'nonlinearity': rectify,
                          'W': init.Orthogonal('relu'),
                          'b': init.Constant(0.01)}),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': N_HIDDEN_2, 'nonlinearity': rectify,
                          'W': init.Orthogonal('relu'),
                          'b': init.Constant(0.01)}),
            (FeaturePoolLayer, {'pool_size': 2}),
            (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
        ]
        args = dict(
            update=adam,
            update_learning_rate=theano.shared(util.float32(START_LR)),
            batch_iterator_train=ResampleIterator(BATCH_SIZE),
            batch_iterator_test=BatchIterator(BATCH_SIZE),
            objective=nn.get_objective(l1=L1, l2=L2),
            eval_size=eval_size,
            custom_scores=[('kappa', metrics.kappa)] if eval_size > 0.0 else None,
            on_epoch_finished=[
                nn.Schedule('update_learning_rate', SCHEDULE),
            ],
            regression=False,
            max_epochs=N_ITER,
            verbose=1,
        )
        est = BlendNet(layers, **args)
        if os.path.exists(final_weights):
            est.load_params_from(str(final_weights))
            print("loaded weights from {}".format(final_weights))

        else:
            est.set_split(files, labels)
            est.fit(X, labels)

            #Persistence
            #est.save_params_to(final_weights)

    return est

"""This function maps the two complementary probabilities in just one,
calculates the area under the ROC curve and plots the curve.
"""
def calc_auc(y_pred_proba, labels, exp_run_folder, classifier, fold):
    
    auc = roc_auc_score(labels, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(labels, y_pred_proba)
    curve_roc = np.array([fpr, tpr])
    dataile_id = open(exp_run_folder+'/data/roc_{}_{}.txt'.format(classifier, fold), 'w+')
    np.savetxt(dataile_id, curve_roc)
    dataile_id.close()
    plt.plot(fpr, tpr, label='ROC curve: AUC={0:0.2f}'.format(auc))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.title('ROC Fold {}'.format(fold))
    plt.legend(loc="lower left")
    plt.savefig(exp_run_folder+'/data/roc_{}_{}.pdf'.format(classifier, fold), format='pdf')
    return auc

@click.command()
@click.option('--cnf', default='configs/vgg_224.py', show_default=True,
              help="Path or name of configuration module.")
@click.option('--exp_run_folder', default=None, show_default=True,
              help="Path to running experiment folder.")
@click.option('--classifier', default='SVM', show_default=True,
              help="Classification method (NN, SVM or RF).")
@click.option('--features_file', default=None, show_default=True,
              help="Read features from specified file.")
@click.option('--n_iter', default=1, show_default=True,
              help="Number of times to fit and average.")
@click.option('--blend_cnf', default='src/utils/blend.yml', show_default=True,
              help="Blending configuration file.")
@click.option('--test_dir', default=None, show_default=True,
              help="Override directory with test set images.")
@click.option('--fold', default='1', show_default=True,
              help="Specify the step of the 10-fold cross-validation (ex.: 1, 2, ..., 10).")
def fit(cnf, exp_run_folder, classifier, features_file, n_iter, blend_cnf, test_dir, fold):

    config = util.load_module(cnf).config
    config.cnf['fold'] = fold                           # <-- used to change the directories for weights_best, weights_epoch and weights_final
    config.cnf['exp_run_folder'] = exp_run_folder

    folds = yaml.load(open('folds/'+data.settings['protocol']+'.yml'))
    f0, f1 = fold.split('x')
    train_list = folds['Fold_' + f0][int(f1)-1]
    test_list  = folds['Fold_' + f0][0 if f1=='2' else 1]

    image_files = data.get_image_files(config.get('train_dir'), train_list)
    names = data.get_names(image_files)
    labels = data.get_labels(names, label_file='folds/'+data.settings['protocol']+'.csv').astype(np.int32)[:, np.newaxis]

    if features_file is not None:
        runs = {'run': [features_file]}
    else:
        runs = {run: [os.path.join(exp_run_folder+'/data/features', f) for f in files] for run, files in yaml.load(open(blend_cnf)).items()}

    scalers = {run: StandardScaler() for run in runs}

    y_preds = []
    y_preds_proba = []
    for i in range(n_iter):
        print("iteration {} / {}".format(i + 1, n_iter))
        for run, files in runs.items():
            files = [ f.replace('f0xf1.npy', '{}.npy'.format(fold)) for f in files ]

            if classifier is None:
                X_test = data.load_features(files, test=True)
                if data.settings['protocol'] != 'protocol3':
                    y_pred_proba = X_test
                    y_proba = []
                    for i in range(0, len(X_test)):
                        y_proba.append(y_pred_proba[i][1]) #using score from the positive
                    y_pred = np.clip(np.round(y_proba), 0, 1).astype(int)
                else:
                    y_pred_proba = est.predict_proba(X)
            else:
                print("fitting features for run {}".format(run))
                X_train = data.load_features(files)
                l2Norm = np.linalg.norm(X_train,axis=1)
                X_train = np.divide(X_train.T,l2Norm).T
                est = estimator(data.settings['protocol'], classifier, X_train.shape[1], image_files, X_train, labels, run, fold, eval_size=0.1)
                open(exp_run_folder+"/best_estimator_fold_{}.txt".format(fold), "w").write(str(est))
                X_test = data.load_features(files, test=True)
                l2Norm = np.linalg.norm(X_test,axis=1)
                X_test = np.divide(X_test.T,l2Norm).T
                if data.settings['protocol'] != 'protocol3':
                    y_pred = est.predict(X_test).ravel()
                    y_pred_proba = est.predict_proba(X_test).ravel()
                    y_proba = []
                    for i in range(0, 2*len(X_test), 2):
                        y_proba.append(y_pred_proba[i+1]) #using score from the positive
                else:
                    y_pred_binary = est.predict(X_test)
                    y_pred = preprocessing.LabelBinarizer().fit([0,1,2])
                    y_pred = y_pred.inverse_transform(y_pred_binary)
                    y_proba = est.predict_proba(X_test)
    
    image_files = data.get_image_files(test_dir or config.get('test_dir'), test_list)
    names = data.get_names(image_files)
    labels = data.get_labels(names, label_file='folds/'+data.settings['protocol']+'.csv').astype(np.int32)[:, np.newaxis]   # , per_patient=per_patient

    image_column = pd.Series(names, name='image')
    labels_column = pd.Series(np.squeeze(labels), name='true')

    level_column = pd.Series(y_pred, name='pred')
    if data.settings['protocol'] != 'protocol3':
        proba_column = pd.Series(y_proba, name='proba')
        predictions = pd.concat([image_column, labels_column, level_column, proba_column], axis=1)
    else:
        proba_label_0 = pd.Series(y_proba[:,0], name='proba_label_0')
        proba_label_1 = pd.Series(y_proba[:,1], name='proba_label_1')
        proba_label_2 = pd.Series(y_proba[:,2], name='proba_label_2')
        predictions = pd.concat([image_column, labels_column, level_column, proba_label_0, proba_label_1, proba_label_2], axis=1)
    

    predictions.to_csv(exp_run_folder+"/ranked_list_fold_{}.csv".format(fold), sep=';')
    
    print("tail of predictions")
    print(predictions.tail())
    acc = len(filter(lambda (l,y) : l == y, zip(labels, y_pred)))/float(len(labels)) 
    print("accuracy: {}".format(acc))
    print("confusion matrix")
    print(confusion_matrix(labels, y_pred))

    if data.settings['protocol'] != 'protocol3':
        auc = calc_auc(y_proba, labels, exp_run_folder, classifier, fold)
        print("AUC: {}".format(auc))
        average_precision = average_precision_score(labels, y_proba)
        print("average precision: {}".format(average_precision))
        c_matrix = confusion_matrix(labels, y_pred)
        print("sensitivity: {}".format(c_matrix[1][1]/(c_matrix[1][1]+ c_matrix[0][1])))
        print("specificity: {}".format(c_matrix[0][0]/(c_matrix[0][0]+ c_matrix[1][0])))
    else:
        y_test = label_binarize(labels, classes=[0, 1, 2])
        auc = roc_auc_score(y_test, y_proba, average='macro')
        print("AUC: {}".format(auc))
        average_precision = average_precision_score(y_test, y_proba, average="macro")
        print("mean average precision: {}".format(average_precision))    

    results = pd.concat([pd.Series(exp_run_folder,name='folder'), pd.Series(fold,name='fold'), pd.Series(auc,name='auc'), pd.Series(average_precision,name='ap'), pd.Series(acc,name='acc')],axis=1)
    with open('results.csv', 'a') as f:
        results.to_csv(f, header=False)

if __name__ == '__main__':
    fit()
