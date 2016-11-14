from datetime import datetime
import pprint
import os

import numpy as np

from util import mkdir

FEATURE_DIR = 'data/features'

class Config(object):
    def __init__(self, layers, cnf=None):
        self.layers = layers
        self.cnf = cnf
        pprint.pprint(cnf)

    def get(self, k, default=None):
        return self.cnf.get(k, default)

    @property
    def weights_epoch(self):
        path = "{}/weights/{}/epochs".format(self.cnf['exp_run_folder'], self.cnf['fold'])
        mkdir(path)
        return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

    @property
    def weights_best(self):
        path = "{}/weights/{}/best".format(self.cnf['exp_run_folder'], self.cnf['fold'])
        mkdir(path)
        return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

    @property
    def weights_file(self):
        path = "{}/weights/{}".format(self.cnf['exp_run_folder'], self.cnf['fold'])
        mkdir(path)
        return os.path.join(path, 'weights.pkl')

    @property
    def retrain_weights_file(self):
        path = "{}/weights/{}/retrain".format(self.cnf['exp_run_folder'], self.cnf['fold'])
        mkdir(path)
        return os.path.join(path, 'weights.pkl')

    @property
    def final_weights_file(self):
        path = "{}/weights/{}".format(self.cnf['exp_run_folder'], self.cnf['fold'])
        mkdir(path)
        return os.path.join(path, 'weights_final.pkl')

    def get_features_fname(self, n_iter, skip=0, test=False):
        path = "{}/{}".format(self.cnf['exp_run_folder'], FEATURE_DIR)
        mkdir(path)
        fname = '{}_{}_mean_iter_{}_skip_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'),  n_iter, skip)
        return os.path.join(path, fname)

    def get_std_fname(self, n_iter, skip=0, test=False):
        path = "{}/{}".format(self.cnf['exp_run_folder'], FEATURE_DIR)
        mkdir(path)
        fname = '{}_{}_std_iter_{}_skip_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'), n_iter, skip)
        return os.path.join(path, fname)

    def save_features(self, X, n_iter, skip=0, test=False):
        np.save(open(self.get_features_fname(n_iter, skip=skip,
                                              test=test), 'wb'), X)

    def save_std(self, X, n_iter, skip=0, test=False):
        np.save(open(self.get_std_fname(n_iter, skip=skip,
                                        test=test), 'wb'), X)

    def load_features(self, test=False):
        return np.load(open(self.get_features_fname(test=test)))

    def get_features_fname_fold(self, n_iter, skip=0, test=False, fold='1x1'):      # <-- Adapted to generate filenames including the folds
        path = "{}/{}".format(self.cnf['exp_run_folder'], FEATURE_DIR)
        mkdir(path)
        fname = '{}_{}_mean_iter_{}_skip_{}_fold_{}.npy'.format(                    # Useful for running with 5x2-fold cross-validation
            self.cnf['name'], ('test' if test else 'train'),  n_iter, skip, fold)
        return os.path.join(path, fname)

    def get_std_fname_fold(self, n_iter, skip=0, test=False, fold='1x1'):
        path = "{}/{}".format(self.cnf['exp_run_folder'], FEATURE_DIR)
        mkdir(path)
        fname = '{}_{}_std_iter_{}_skip_{}_fold_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'), n_iter, skip, fold)
        return os.path.join(path, fname)

    def save_features_fold(self, X, n_iter, skip=0, test=False, fold='1x1'):
        np.save(open(self.get_features_fname_fold(n_iter, skip=skip,
                                              test=test, fold=fold), 'wb'), X)

    def save_std_fold(self, X, n_iter, skip=0, test=False, fold='1x1'):
        np.save(open(self.get_std_fname_fold(n_iter, skip=skip,
                                        test=test, fold=fold), 'wb'), X)

##
# The functions weights_epoch, weights_best, weights_file, retrain_weights_file
# and final_weights_file were adapted to put the weights into a directory that
# references the current fold
##
