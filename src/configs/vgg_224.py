import numpy as np
import pickle as pkl
import sys
sys.path.insert(0, 'src/utils/')
sys.path.insert(0, 'folds/')

from config import Config
from layers import *
from lasagne.nonlinearities import softmax
import data

cnf = {
	'name': __name__.split('.')[-1],
	'w': 224,
	'h': 224,
	'train_dir': data.settings['images_dir'],
	'test_dir': data.settings['images_dir'],
	'batch_size_train': 48,
	'batch_size_test': 16,
	'balance_weights': np.array(data.settings['balance_weights']),
	'balance_ratio': 0.975,
	'final_balance_weights':  data.settings['final_balance_weights'],
	'aug_params': {
		'zoom_range': (1 / 1.30, 1.30),
	    'rotation_range': (0, 360),
	    'shear_range': (-30, 30),
	    'translation_range': (-30, 30),
	    'do_flip': True,
	    'allow_stretch': True,
	},
	'sigma': 0.5,
	'schedule': {
		0: 0.001,
		80: 'stop',
		#201: 'stop',
	},
}

num_outputs = 2
if data.settings['train_retina']:
	num_outputs = 5
elif data.settings['protocol'] == 'protocol3':
	num_outputs = 3

#VGG-M
layers = [
	(InputLayer, {'shape': (None, 3, 224, 224)}),
	(Conv2DLayer, conv_params(96, filter_size=(7, 7), stride=(2, 2), border_mode=0, nonlinearity=rectify)),
	(LocalResponseNormalization2DLayer, {'alpha': 0.0001, 'k': 2, 'beta': 0.75, 'n': 5}),
	(MaxPool2DLayer, pool_params()),
	(Conv2DLayer, conv_params(256, filter_size=(5, 5), stride=(2, 2), border_mode=1, nonlinearity=rectify)),
	(LocalResponseNormalization2DLayer, {'alpha': 0.0001, 'k': 2, 'beta': 0.75, 'n': 5}),
	(MaxPool2DLayer, pool_params(pool_size=2)),
	(Conv2DLayer, conv_params(512, filter_size=(3, 3), stride=(1, 1), border_mode=1, nonlinearity=rectify)),
	(Conv2DLayer, conv_params(512, filter_size=(3, 3), stride=(1, 1), border_mode=1, nonlinearity=rectify)),
	(Conv2DLayer, conv_params(512, filter_size=(3, 3), stride=(1, 1), border_mode=1, nonlinearity=rectify)),
	(MaxPool2DLayer, pool_params()),
	(DenseLayer, dense_params(4096)),
	(DropoutLayer, {'p': 0.5}),
	(DenseLayer, dense_params(4096)),
	(DropoutLayer, {'p': 0.5}),
	(DenseLayer, dense_params(num_outputs, nonlinearity = softmax))
]

n = 64

#VGG-16
#layers = [
#    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
#    (Conv2DLayer, conv_params(n, border_mode=1)),
#    (Conv2DLayer, conv_params(n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (Conv2DLayer, conv_params(2 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(2 * n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (Conv2DLayer, conv_params(4 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(4 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(4 * n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (DenseLayer, dense_params(4096)),
#    (DropoutLayer, {'p': 0.5}),
#    (DenseLayer, dense_params(4096)),
#    (DropoutLayer, {'p': 0.5}),
#    (DenseLayer, dense_params(num_outputs, nonlinearity = softmax))
#]

#VGG-19
#layers = [
#    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
#    (Conv2DLayer, conv_params(n, border_mode=1)),
#    (Conv2DLayer, conv_params(n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (Conv2DLayer, conv_params(2 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(2 * n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (Conv2DLayer, conv_params(4 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(4 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(4 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(4 * n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (Conv2DLayer, conv_params(8 * n, border_mode=1)),
#    (MaxPool2DLayer, pool_params(pool_size=2)),
#    (DenseLayer, dense_params(4096)),
#    (DropoutLayer, {'p': 0.5}),
#    (DenseLayer, dense_params(4096)),
#    (DropoutLayer, {'p': 0.5}),
#    (DenseLayer, dense_params(num_outputs, nonlinearity = softmax))
#]
config = Config(layers=layers, cnf=cnf)
