KNOWLEDGE TRANSFER FOR MELANOMA SCREENING WITH DEEP LEARNING
=========================

This repository contains the implementation of simple and double transfer learning schemes to classify melanoma images, using pre-trained generic models like ImageNet and training specific models with a diabetic retinopaty dataset. We have compared them in the paper "Knowledge Transfer for Melanoma Screening with Deep Learning" (published at ISBI 2017).

You can find more details about the concepts and parametrization of each technique at the paper. The codes available in this repository are already parametrized according to the details provided in the final text.

The source codes of this repository is a branch of [team o_O for the diabetic retinopath competition](https://github.com/sveitser/kaggle_diabetic). Some Bash and Python scripts were modified to adapt the codes for melanoma screening problem.

If you use this work in any academic/official context, please cite us. The relevant references are below.

TABLE OF CONTENTS
------------------

 - Dependencies
 - Repository contents
 - Getting Prepared
 - Running

DEPENDENCIES
-------------

Note: we assume that the codes available here will be run on an Unix OS. You will only need to download [ImageMagick](http://www.imagemagick.org/script/index.php) and Python (we strongly suggest to use the Anaconda package).

After dependencies are installed, run the command in the root folder:

	pip install -r requirements.txt

That's it!

REPOSITORY CONTENTS
--------------------

	folds: 					directory containing the fold files of each validation protocol.
	src:
		configs:				this folder contains the network layers configuration, data augmentation parameters, etc.
		utils:					utils folder.
		train_nn.py:				main training/fine tuning script.
		transform.py:				feature extractor script.
		blend.py:				SVM classifier and results script.
	run_all.sh 				script to run all the experiments at once.
	main.sh					main script to run the experiments.
	fetch.sh				script to fetch pre-trained imagenet vgg-m model and resize melanoma and retina raw images.
	convertEDRAAtlas.sh		script to remove black borders on original EDRA database.
	settings.py:				file that generate basic settings needed for each experiment.
	requirements.txt: 			file with required dependencies to run the experiments.
	README.md:	 			this file.

GETTING PREPARED
--------
To run the experiments, you will need to download the pre-trained imagenet vgg-m model and resize all melanoma and retina images.

1) To download the pre-trained VGG-M ImageNet model and convert it from .mat to Nolearn readable .pkl file, run the following command:

	./fetch.sh --imagenet

This will save the transformed weights in datasets/imagenet/vggm.pkl

2) To resize melanoma images from the EDRA dataset, run the following commands:

	./convertEDRAAtlas.sh <EDRAS CD-ROM Image path>  <target path>
	./fetch.sh --melanoma --infolder <converted images path> --outfolder <resized melanoma images path>

3) To crop and resize retina images from the Kaggle dataset, please create a Kaggle account and go to:

	https://www.kaggle.com/c/diabetic-retinopathy-detection

In the "Data" tab, download all the zipped train dataset (32.68Gb total), unzip it and get all the retina images in a flat folder. Run:

	./fetch.sh --retina --infolder <flat retina images path> --outfolder <resized retina images path>

Now that you have the ImageNet pre-trained model and both melanoma and retina datasets in two separate flat folders, please go to settings.py and update the "Path to resized images" section with the output folder you used in step 2 and 3.

RUNNING
--------
The paper describes three protocols numbered from 1 to 3 with six experiments each, labeled A to F. To run an experiment:

	./main.sh -p [PROTOCOL] -e [EXPERIMENT]

where:

[PROTOCOL] = 'protocol1' or 'protocol2' or 'protocol3'
[EXPERIMENT] = 'A' to 'F', where:

	A: Training and testing a network from scratch with melanoma images;
	B: Training a network from scratch with diabetic retinopathy images,
		transfer learning for melanoma *without* fine tuning;
	C: Training a network from scratch with diabetic retinopathy images,
		transfer learning for melanoma *with* fine tuning;
	D: Uploading a pretrained network with ImageNet images, transfer learning
		for melanoma *without* fine tuning;
	E: Uploading a pretrained network with ImageNet images, transfer learning
		for melanoma *with* fine tuning;
	F: Uploading a pretrained network with ImageNet images, transfer learning
		for diabetic retinopathy *with* fine tuning then transferring again
		for melanoma images *with* fine tuning;


OR run all the experiments at once using the run all script:
	./run_all.sh


REFERENCES
---------

The main reference for this work is:

A. Menegola, M. Fornaciali, R. Pires, F. V. Bittencourt, S. Avila and E. Valle.
"Knowledge transfer for melanoma screening with deep learning," 2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017), Melbourne, VIC, 2017, pp. 297-300.
DOI: [10.1109/ISBI.2017.7950523](http://dx.doi.org/10.1109/ISBI.2017.7950523)

The last preprint (essentially identitcal to the published version) is linked at arXiv:
[Knowledge Transfer for Melanoma Screening with Deep Learning](https://arxiv.org/abs/1703.07479)

A rough preliminary version of this paper is also at arXiv:
[Towards Automated Melanoma Screening: Exploring Transfer Learning Schemes](http://128.84.21.199/pdf/1609.01228).

Please prefer to cite the published version (first item above) instead of the arXiv preprints.
