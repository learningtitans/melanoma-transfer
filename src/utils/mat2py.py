import numpy as np
from numpy import rollaxis, swapaxes
from collections import OrderedDict
import scipy.io
import pickle as pickle
mat = scipy.io.loadmat('datasets/imagenet/imagenet-vgg-m.mat')
#mat = scipy.io.loadmat('datasets/imagenet/imagenet-vgg-verydeep-16.mat')

def rolling(roll_me):
    a = swapaxes(roll_me, 3, 0)
    a = swapaxes(a, 1, 2)
    a = swapaxes(a, 2, 3)
    return a

#vggm
conv2ddnn1 = [np.array(rolling(mat['layers'][0][0][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][0][0][0][2][0][1]),dtype='float32')]
conv2ddnn4 = [np.array(rolling(mat['layers'][0][4][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][4][0][0][2][0][1]),dtype='float32')]
conv2ddnn7 = [np.array(rolling(mat['layers'][0][8][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][8][0][0][2][0][1]),dtype='float32')]
conv2ddnn8 = [np.array(rolling(mat['layers'][0][10][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][10][0][0][2][0][1]),dtype='float32')]
conv2ddnn9 = [np.array(rolling(mat['layers'][0][12][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][12][0][0][2][0][1]),dtype='float32')]
dense11 = [np.array(np.reshape(rollaxis(mat['layers'][0][15][0][0][2][0][0], 2),(18432,4096)),dtype='float32'), np.array(np.squeeze(mat['layers'][0][15][0][0][2][0][1]),dtype='float32')]
dense13 = [np.array(np.squeeze(mat['layers'][0][17][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][17][0][0][2][0][1]),dtype='float32')]
dense15 = [np.array(np.squeeze(mat['layers'][0][19][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][19][0][0][2][0][1]),dtype='float32')]

vggm = (('input0',[]),
		('conv2ddnn1',conv2ddnn1),
		('localresponsenormalization2d2',[]),
		('maxpool2ddnn3',[]),
		('conv2ddnn4',conv2ddnn4),
		('localresponsenormalization2d5',[]),
		('maxpool2ddnn6',[]),
		('conv2ddnn7',conv2ddnn7),
		('conv2ddnn8',conv2ddnn8),
		('conv2ddnn9',conv2ddnn9),
		('maxpool2ddnn10',[]),
		('dense11',dense11),
		('dropout12',[]),
		('dense13',dense13),
		('dropout14',[]),
		('dense15',dense15))


#vggm-no-local-response
#vggm = (('input0',[]),
#		('conv2ddnn1',conv2ddnn1),
#		#('localresponsenormalization2d2',[]),
#		('maxpool2ddnn2',[]),
#		('conv2ddnn3',conv2ddnn4),
#		#('localresponsenormalization2d5',[]),
#		('maxpool2ddnn4',[]),
#		('conv2ddnn5',conv2ddnn7),
#		('conv2ddnn6',conv2ddnn8),
#		('conv2ddnn7',conv2ddnn9),
#		('maxpool2ddnn8',[]),
#		('dense9',dense11),
#		('dropout10',[]),
#		('dense11',dense13),
#		('dropout12',[]),
#		('dense13',dense15))

#vgg16
#conv2ddnn1 = [np.array(rolling(mat['layers'][0][0][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][0][0][0][2][0][1]),dtype='float32')]
#conv2ddnn2 = [np.array(rolling(mat['layers'][0][2][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][2][0][0][2][0][1]),dtype='float32')]
#conv2ddnn4 = [np.array(rolling(mat['layers'][0][5][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][5][0][0][2][0][1]),dtype='float32')]
#conv2ddnn5 = [np.array(rolling(mat['layers'][0][7][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][7][0][0][2][0][1]),dtype='float32')]
#conv2ddnn7 = [np.array(rolling(mat['layers'][0][10][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][10][0][0][2][0][1]),dtype='float32')]
#conv2ddnn8 = [np.array(rolling(mat['layers'][0][12][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][12][0][0][2][0][1]),dtype='float32')]
#conv2ddnn9 = [np.array(rolling(mat['layers'][0][14][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][14][0][0][2][0][1]),dtype='float32')]
#conv2ddnn11 = [np.array(rolling(mat['layers'][0][17][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][17][0][0][2][0][1]),dtype='float32')]
#conv2ddnn12 = [np.array(rolling(mat['layers'][0][19][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][19][0][0][2][0][1]),dtype='float32')]
#conv2ddnn13 = [np.array(rolling(mat['layers'][0][21][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][21][0][0][2][0][1]),dtype='float32')]
#conv2ddnn15 = [np.array(rolling(mat['layers'][0][24][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][24][0][0][2][0][1]),dtype='float32')]
#conv2ddnn16 = [np.array(rolling(mat['layers'][0][26][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][26][0][0][2][0][1]),dtype='float32')]
#conv2ddnn17 = [np.array(rolling(mat['layers'][0][28][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][28][0][0][2][0][1]),dtype='float32')]
#dense19 = [np.array(np.reshape(rollaxis(mat['layers'][0][31][0][0][2][0][0], 2),(25088,4096)),dtype='float32'), np.array(np.squeeze(mat['layers'][0][31][0][0][2][0][1]),dtype='float32')]
#dense21 = [np.array(np.squeeze(mat['layers'][0][33][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][33][0][0][2][0][1]),dtype='float32')]
#dense23 = [np.array(np.squeeze(mat['layers'][0][35][0][0][2][0][0]),dtype='float32'), np.array(np.squeeze(mat['layers'][0][35][0][0][2][0][1]),dtype='float32')]
#
#vggm = (('input0',[]),
#		('conv2ddnn1',conv2ddnn1),
#		('conv2ddnn2',conv2ddnn2),
#		('maxpool2ddnn3',[]),
#		('conv2ddnn4',conv2ddnn4),
#		('conv2ddnn5',conv2ddnn5),
#		('maxpool2ddnn6',[]),
#		('conv2ddnn7',conv2ddnn7),
#		('conv2ddnn8',conv2ddnn8),
#		('conv2ddnn9',conv2ddnn9),
#		('maxpool2ddnn10',[]),
#		('conv2ddnn11',conv2ddnn11),
#		('conv2ddnn12',conv2ddnn12),
#		('conv2ddnn13',conv2ddnn13),
#		('maxpool2ddnn14',[]),
#		('conv2ddnn15',conv2ddnn15),
#		('conv2ddnn16',conv2ddnn16),
#		('conv2ddnn17',conv2ddnn17),
#		('maxpool2ddnn18',[]),
#		('dense19',dense19),
#		('dropout20',[]),
#		('dense21',dense21),
#		('dropout22',[]),
#		('dense23',dense23))

vggm = OrderedDict(vggm)

mat = None

print("Writing to .pkl file...")
pickle.dump( vggm, open( "datasets/imagenet/vgg16.pkl", "wb" ) )
