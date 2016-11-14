"""Conv Nets training script."""
import click
import yaml
import pickle as pkl

import sys
sys.path.insert(0, 'src/utils/')
sys.path.insert(0, 'src/configs/')

import data
import util
import nn

import numpy as np
np.random.seed(9)

@click.command()
@click.option('--cnf', default='configs/vgg_224.py', show_default=True,
              help='Path or name of configuration module.')
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
@click.option('--exp_run_folder', default=None, show_default=True,
              help="Path to running experiment folder.")
@click.option('--train_retina', default=None, show_default=True,
              help="Flag to train retina.")
@click.option('--fold', default='1x1', show_default=True,
              help="Specify the step of the 5x2-fold cross-validation (ex.: 1x1, 1x2, ..., 5x2).")

def main(cnf, weights_from, fold, exp_run_folder, train_retina):
    config = util.load_module(cnf).config
    config.cnf['fold'] = fold                           # <-- used to change the directories for weights_best, weights_epoch and weights_final
    config.cnf['exp_run_folder'] = exp_run_folder
    protocol = data.settings['protocol']

    if train_retina != 'train_retina':
        folds = yaml.load(open('folds/'+protocol+'.yml'))
        f0, f1 = fold.split('x')
        train_list = folds['Fold_' + f0][int(f1)-1]
        files = data.get_image_files(config.get('train_dir'), train_list)
    else:
        files = data.get_image_files(config.get('train_dir'))

    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)

    names = data.get_names(files)
    labels = data.get_labels(names, label_file='folds/'+protocol+'.csv').astype(np.int32)
    net = nn.create_net(config)

    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights, starting from scratch")


    #Print layerinfo
    print("## Layer information")
    import nolearn
    layer_info = nolearn.lasagne.PrintLayerInfo()
    print(layer_info._get_greeting(net))
    layer_info, legend = layer_info._get_layer_info_conv(net)
    print(layer_info)
    print(legend)
    print("fitting ...")
    net.fit(files, labels)

if __name__ == '__main__':
    main()
