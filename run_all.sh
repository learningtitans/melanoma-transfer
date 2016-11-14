#!/usr/bin/env bash

#!/bin/bash

#
#       This script can be used to generate the 2nd place solution
#       of team o_O for the diabetic retinopath competition.
#

# terminate on error
set -e

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
	python settings.py --protocol $protocol --base_train_params melanoma --dataset melanoma
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
		python settings.py --protocol retina --base_train_params retina --dataset retina
		ret_folder=$"datasets/retina/exp_B_and_C" && mkdir -p $ret_folder
		python src/train_nn.py --exp_run_folder $ret_folder --train_retina train_retina --fold $FOLD
		BEST_VALID_WEIGHTS="$(ls -t $ret_folder/weights/$FOLD/best/ | head -n 1)" && cp $ret_folder/weights/$FOLD/best/$BEST_VALID_WEIGHTS $ret_folder/weights/ && mv $ret_folder/weights/$BEST_VALID_WEIGHTS $ret_folder/weights/retina.pkl
	fi
	python settings.py --protocol $protocol --base_train_params retina --dataset melanoma
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from datasets/retina/exp_B_and_C/weights/retina.pkl  --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

experiment_C() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	if [[ ! -f datasets/retina/exp_B_and_C/weights/retina.pkl ]]; then
		echo "$(tput setaf 2)Training with retina database ...$(tput sgr 0)"
		python settings.py --protocol retina --base_train_params retina --dataset retina
		ret_folder=$"datasets/retina/exp_B_and_C" && mkdir -p $ret_folder
		python src/train_nn.py --exp_run_folder $ret_folder --train_retina train_retina --fold $FOLD
		BEST_VALID_WEIGHTS="$(ls -t $ret_folder/weights/$FOLD/best/ | head -n 1)" && cp $ret_folder/weights/$FOLD/best/$BEST_VALID_WEIGHTS $ret_folder/weights/ && mv $ret_folder/weights/$BEST_VALID_WEIGHTS $ret_folder/weights/retina.pkl
	fi
	echo "$(tput setaf 2)Fold $FOLD: Fine tuning retina with melanoma database ...$(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params retina --dataset melanoma
	python src/train_nn.py --weights_from datasets/retina/exp_B_and_C/weights/retina.pkl --exp_run_folder $new_exp --fold $FOLD
	BEST_VALID_WEIGHTS="$(ls -t $new_exp/weights/$FOLD/best/ | head -n 1)"
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from $new_exp/weights/$FOLD/best/$BEST_VALID_WEIGHTS  --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

experiment_D() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params imagenet --dataset melanoma
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from datasets/imagenet/vggm.pkl --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

experiment_E() {
	echo "$(tput setaf 2)The results for this experiment are being saved on folder $new_exp $(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params imagenet --dataset melanoma
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
		python settings.py --protocol retina --base_train_params imagenet --dataset retina
		echo "$(tput setaf 2)Fine tuning imagenet with retina database ...$(tput sgr 0)"
		ret_folder=$"datasets/retina/exp_F" && mkdir -p $ret_folder
		python src/train_nn.py --weights_from datasets/imagenet/vggm.pkl --exp_run_folder $ret_folder --train_retina train_retina --fold $FOLD
		BEST_VALID_WEIGHTS="$(ls -t $ret_folder/weights/$FOLD/best/ | head -n 1)" && cp $ret_folder/weights/$FOLD/best/$BEST_VALID_WEIGHTS $ret_folder/weights/ && mv $ret_folder/weights/$BEST_VALID_WEIGHTS $ret_folder/weights/retina.pkl
	fi
	echo "$(tput setaf 2)Fold $FOLD: Fine tuning imagenet+retina with melanoma database ...$(tput sgr 0)"
	python settings.py --protocol $protocol --base_train_params imagenet --dataset melanoma
	python src/train_nn.py --weights_from datasets/retina/exp_F/weights/retina.pkl --exp_run_folder $new_exp --fold $FOLD
	BEST_VALID_WEIGHTS="$(ls -t $new_exp/weights/$FOLD/best/ | head -n 1)"
	echo "$(tput setaf 2)Fold $FOLD: Extracting features ...$(tput sgr 0)"
	python src/transform.py --exp_run_folder $new_exp --train --test --n_iter 1 --weights_from $new_exp/weights/$FOLD/best/$BEST_VALID_WEIGHTS  --fold $FOLD
	echo "$(tput setaf 2)Fold $FOLD: Classifying ...$(tput sgr 0)"
	python src/blend.py --exp_run_folder $new_exp --fold $FOLD --classifier 'SVM'
}

run_experiment() {
	declare -a protocols=("protocol1" "protocol2" "protocol3")
	declare -a exps=("A" "B" "C" "D" "E" "F")

	for protocol in "${protocols[@]}"
	do
		for experiment in "${exps[@]}"
		do
			new_exp="results/$protocol$experiment"
			mkdir -p $new_exp
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
		done
	done
	echo "$(tput setaf 2)Step 1. Executing experiment $experiment ...: DONE! "
}

#####################################################################################
#																					#
#									MAIN PROGRAM									#
#																					#
#####################################################################################

echo "$(tput setaf 2)WARNING: Running all the experiments can take a relative long time (it depends on who is at light speed, you or your computer).$(tput sgr 0)"
# Main pipeline
run_experiment

# End of baseline script
echo ""
echo "$(tput setaf 2)F I N I S H E D! $(tput sgr 0)"