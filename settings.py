import numpy as np
import pickle as pkl
import click
import sys
sys.path.insert(0, 'src/utils/')
sys.path.insert(0, 'folds/')

### Path to resized images
melanoma_dir = '/path/to/melanoma/dir/' #path to resized melanoma images folder
retina_dir = '/path/to/retina/dir/' #path to resized retina images folder
isic_dir = '/path/to/isic/dir/' #path to resized isic images folder

### Melanoma parameters
melanoma_balance_weights = np.array([1.4595, 3.1758]) # set of resampling weights that yields balanced classes
isic_balance_weights = np.array([1.2405, 5.1572])
melanoma_prot3_balance_weights = np.array([23.2142, 3.6792, 1.4595])
melanoma_std = np.array([ 49.33694864,  56.56751696,  59.42470697], dtype=np.float32) # channel standard deviations
melanoma_mean = np.array([ 198.19906616, 170.38525391, 155.49664307], dtype=np.float32) # channel means
melanoma_U = np.array([[-0.51556925,  0.77283555,  0.35053246],
 				 [-0.60600832, -0.04596143, -0.78214996],
 				 [-0.60304644, -0.6169369,   0.49839384]] ,dtype=np.float32)
melanoma_EV = np.array([ 102.92342015, 25.71846177, 11.24928707], dtype=np.float32)

### Retina parameters
retina_balance_weights = np.array([1.360945,  14.378223, 6.637566, 40.235967, 49.612994]) # set of resampling weights that yields balanced classes
retina_std = np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32) # channel standard deviations
retina_mean = np.array([123.6591, 116.7663, 103.9318], dtype=np.float32) # channel means
retina_U = np.array([[-0.56543481, 0.71983482, 0.40240142],
			         [-0.5989477, -0.02304967, -0.80036049],
			         [-0.56694071, -0.6935729, 0.44423429]] ,dtype=np.float32)
retina_EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)

### Imagenet parameters
imagenet_mean = np.array([116.7663, 123.6591, 103.9318], dtype=np.float32)

@click.command()
@click.option('--protocol', default=None)
@click.option('--base_train_params', default=None)
@click.option('--dataset', default=None)

def main(protocol, base_train_params, dataset):
	settings = {}
	settings['protocol'] = protocol
	settings['train_retina'] = False
	if protocol == 'retina':
		settings['train_retina'] = True

	if dataset == 'retina':
		settings['images_dir'] = retina_dir
		settings['balance_weights'] = retina_balance_weights
		settings['final_balance_weights'] = np.array([1, 2, 2, 2, 2], dtype=float)
	elif dataset == 'isic':
		settings['images_dir'] = isic_dir
		settings['balance_weights'] = isic_balance_weights
		settings['final_balance_weights'] = np.array([1, 5], dtype=float)
	else:
		settings['images_dir'] = melanoma_dir
		if settings['protocol'] == 'protocol3':
			settings['balance_weights'] = melanoma_prot3_balance_weights
			settings['final_balance_weights'] = np.array([23, 3, 1], dtype=float)
		else:
			settings['balance_weights'] = melanoma_balance_weights
			settings['final_balance_weights'] = np.array([1, 3], dtype=float)

	if base_train_params == 'imagenet':
		settings['std'] = None
		settings['mean'] = imagenet_mean
		settings['u'] = None
		settings['ev'] = None
	elif base_train_params == 'retina':
		settings['std'] = retina_std
		settings['mean'] = retina_mean
		settings['u'] = retina_U
		settings['ev'] = retina_EV
	elif base_train_params == 'melanoma':
		settings['std'] = melanoma_std
		settings['mean'] = melanoma_mean
		settings['u'] = melanoma_U
		settings['ev'] = melanoma_EV

	with open('src/configs/settings.pkl', 'wb') as f:
  		pkl.dump(settings, f)

if __name__ == '__main__':
    main()
