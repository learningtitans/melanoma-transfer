#!/usr/bin/env bash

#!/bin/bash

#
#       This script can be used to generate the 2nd place solution
#       of team o_O for the diabetic retinopath competition.
#

# terminate on error
set -e

# ARGUMENTS. No need to change this. 
dataset=""
experiment=""
path_matlab=""
# ARGUMENTS END.

# Usage
show_help() {
	echo "USAGE: main.sh -d [DATASET] -p [PROTOCOL] -e [EXPERIMENT]"
	echo "WHERE: "
	echo "	[DATASET] 	: 'melanoma', 'isic' or 'retina'"
	echo "	[PROTOCOL] 	: 'protocol1' to 'protocol3', as detailed at the paper "
	echo "	[EXPERIMENT] 	: 'A' to 'F', as detailed at the paper "
	echo ""
	echo "	ATTENTION! This script assumes that the dependencies are settled in the path."
	echo "	-> See README file for the list of dependencies;"
}

# Check if all needed arguments were informed (not empty)
check_arguments() {
	if [ -z $dataset ] || [ -z $protocol ] || [ -z $experiment ] ;
	 	then
		show_help
		exit 1
	fi
}

# Input arguments
show_arguments() {
	echo "ARGUMENTS: "
	echo "	Dataset     	=	$dataset"
	echo "	Protocol		=	$protocol"
	echo "	Experiment		=	$experiment"
}

# Setup: create directories as needed
create_directories() {
	echo "$(tput setaf 2)Creating auxiliary directories...$(tput sgr 0)"
	mkdir -p results
	new_exp="results/"$experiment"_"$(date +%Y-%m-%d:%H:%M:%S)
	mkdir -p $new_exp
	echo "$(tput setaf 2)Creating auxiliary directories...: DONE!$(tput sgr 0)"
}

experiment_A() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params melanoma --dataset $dataset
	echo "$(tput setaf 2)Fold $FOLD: Training with melanoma database ...$(tput sgr 0)"
	python src/train_nn.py --exp_run_folder $new_exp --fold $FOLD
	BEST_VALID_WEIGHTS="$(ls -t $new_exp/weights/$FOLD/best/ | head -n 1)"
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from $new_exp/weights/$FOLD/best/$BEST_VALID_WEIGHTS  --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

experiment_B() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	if [[ ! -f datasets/retina/exp_B_and_C/weights/retina.pkl ]]; then
		echo "$(tput setaf 2)Training with retina database ...$(tput sgr 0)"
		python settings.py --protocol retina --base_train_params retina --dataset $dataset
		ret_folder=$"datasets/retina/exp_B_and_C" && mkdir -p $ret_folder
		python src/train_nn.py --exp_run_folder $ret_folder --train_retina train_retina --fold $FOLD
		BEST_VALID_WEIGHTS="$(ls -t $ret_folder/weights/$FOLD/best/ | head -n 1)" && cp $ret_folder/weights/$FOLD/best/$BEST_VALID_WEIGHTS $ret_folder/weights/ && mv $ret_folder/weights/$BEST_VALID_WEIGHTS $ret_folder/weights/retina.pkl
	fi
	python settings.py --protocol $protocol --base_train_params retina --dataset $dataset
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from datasets/retina/exp_B_and_C/weights/retina.pkl  --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

experiment_C() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	if [[ ! -f datasets/retina/exp_B_and_C/weights/retina.pkl ]]; then
		echo "$(tput setaf 2)Training with retina database ...$(tput sgr 0)"
		python settings.py --protocol retina --base_train_params retina --dataset $dataset
		ret_folder=$"datasets/retina/exp_B_and_C" && mkdir -p $ret_folder
		python src/train_nn.py --exp_run_folder $ret_folder --train_retina train_retina --fold $FOLD
		BEST_VALID_WEIGHTS="$(ls -t $ret_folder/weights/$FOLD/best/ | head -n 1)" && cp $ret_folder/weights/$FOLD/best/$BEST_VALID_WEIGHTS $ret_folder/weights/ && mv $ret_folder/weights/$BEST_VALID_WEIGHTS $ret_folder/weights/retina.pkl
	fi
	echo "$(tput setaf 2)Fold $FOLD: Fine tuning retina with melanoma database ...$(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params retina --dataset $dataset
	python src/train_nn.py --weights_from datasets/retina/exp_B_and_C/weights/retina.pkl --exp_run_folder $new_exp --fold $FOLD
	BEST_VALID_WEIGHTS="$(ls -t $new_exp/weights/$FOLD/best/ | head -n 1)"
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from $new_exp/weights/$FOLD/best/$BEST_VALID_WEIGHTS  --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

experiment_D() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params imagenet --dataset $dataset
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from datasets/imagenet/vggm.pkl --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

experiment_E() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params imagenet --dataset $dataset
	echo "$(tput setaf 2)Fold $FOLD: Fine tuning imagenet with melanoma database ...$(tput sgr 0)"
	python src/train_nn.py --weights_from datasets/imagenet/vggm.pkl --exp_run_folder $new_exp --fold $FOLD
	BEST_VALID_WEIGHTS="$(ls -t $new_exp/weights/$FOLD/best/ | head -n 1)"
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from $new_exp/weights/$FOLD/best/$BEST_VALID_WEIGHTS  --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

experiment_F() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	if [[ ! -f datasets/retina/exp_F/weights/retina.pkl ]]; then
		python settings.py --protocol retina --base_train_params imagenet --dataset $dataset
		echo "$(tput setaf 2)Fine tuning imagenet with retina database ...$(tput sgr 0)"
		ret_folder=$"datasets/retina/exp_F" && mkdir -p $ret_folder
		python src/train_nn.py --weights_from datasets/imagenet/vggm.pkl --exp_run_folder $ret_folder --train_retina train_retina --fold $FOLD
		BEST_VALID_WEIGHTS="$(ls -t $ret_folder/weights/$FOLD/best/ | head -n 1)" && cp $ret_folder/weights/$FOLD/best/$BEST_VALID_WEIGHTS $ret_folder/weights/ && mv $ret_folder/weights/$BEST_VALID_WEIGHTS $ret_folder/weights/retina.pkl
	fi
	echo "$(tput setaf 2)Fold $FOLD: Fine tuning imagenet+retina with melanoma database ...$(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params imagenet --dataset $dataset
	python src/train_nn.py --weights_from datasets/retina/exp_F/weights/retina.pkl --exp_run_folder $new_exp --fold $FOLD
	BEST_VALID_WEIGHTS="$(ls -t $new_exp/weights/$FOLD/best/ | head -n 1)"
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from $new_exp/weights/$FOLD/best/$BEST_VALID_WEIGHTS  --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

run_experiment() {
	echo "$(tput setaf 2)Step 1. Executing experiment $experiment ...$(tput sgr 0)"
	# define the fold
	for i in `seq 1 5`
	do
  		for j in `seq 1 2`
  		do
    		FOLD=$i"x"$j
			case "$experiment" in
				A) experiment_A ;;
				B) experiment_B ;;
				C) experiment_C ;;
				D) experiment_D ;;
				E) experiment_E ;;
				F) experiment_F ;;
			esac
  		done
	done
	echo "$(tput setaf 2)Step 1. Executing experiment $experiment ...: DONE! "
}

#####################################################################################
#																					#
#									MAIN PROGRAM									#
#																					#
#####################################################################################

# If there is no enough arguments, tells the user how to use this script
OPTIND=1         
if [ $# -lt 6 ]; then 
	show_help 
	exit 1 
fi

# Parse the arguments
while getopts "d:p:e:" opt; do
    case "$opt" in
   	d)  dataset=$OPTARG ;;
	p)  protocol=$OPTARG ;;
	e)  experiment=$OPTARG ;;
	esac
done

# Before starting, check arguments
check_arguments

# Setup
show_arguments
create_directories

# Main pipeline
run_experiment

# End of baseline script
echo ""
echo "$(tput setaf 2)F I N I S H E D! $(tput sgr 0)"
