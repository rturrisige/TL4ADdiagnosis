# TL4ADdiagnosis

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pytorch](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

This repository enables the use of Transfer Learning (TL) to enhance Alzhaimer's Disease (AD) diagnosis based on images.  
We recommand to use ADnet (available at https://github.com/rturrisige/3D_CNN_pretrained_model) or its variants in case of MRI dataset. 
Otherwise, ResNet models pre-trained on ImageNet and adapted to 3D processing can be used. 


## General Approach (GA)

- GA_feature_extraction.py: uses ADnet or ResNet-based models as feature extractor for 3D MRI scans
- GA_classification.py: load the extracted features and performs AD diagnosis based on SVM, Knn, and RF classifiers
- GA_utilities.py: contains utility functions for feature extraction and classification. 
  The function 'feature_data_preparation' assumes data being grouped in AD and CN sets, labelled as 2 and 0 respecitvely.
  Modified this function for customized usage.
- GA_plot_functions.py: contains useful functions to plot results.

### Usage example

	DATA_DIR=/home/mydata
    FEATURE_DIR=/home/mydata/features
	SAVER_DIR=/home/GA_Results
    MODEL=ResNet101
    AUGMENTATION=True

	python GA_feature_extraction.py --data_dir $DATA_DIR \
					--saver_dir $FEATURE_DIR \
                                    --model $MODEL


	python GA_classification.py --model $MODEL \
				 --feature_dir FEATURE_DIR \
				 --augmentation $AUGMENTATION \
				 --saver_dir $SAVER_DIR 

## Deep Approach (DA)

- DA_data_augmentation.py: generates augmented data by applying affine transformations to 3D MRI images
- DA_data_loading.py: contains utility functions for loading data. Please, modify "my_data_list" and "my_augmented_data_finder" for custemized usage
- DA_utilities.py: contains utility functions for model fine-tuning and evaluation
- DA_finetuning.py: load a pre-trained model and perform fine-tuning for AD diagnosis
- DA_finetuning_configuration.py: contains the hyper-parameters configuration for fine-tuning (e.g., the number of training epochs) 
- DA_plot_functions.py: contains useful functions to plot results.

### Usage example

	DATA_DIR=/home/mydata
	AUGMENTATION_DIR=/home/mydata/augmented_data
	SAVER_DIR=/home/DA_Results
	MODEL=ResNet101

	python DA_data_augmentation.py --data_dir $DATA_DIR \
					--saver_dir $AUGMENTATION_DIR

	python DA_fine_tuning.py --model $MODEL \
				 --data_dir $DATA_DIR \
				 --augmentation_dir $AUGMENTATION_DIR \
				 --saver_dir $SAVER_DIR 
						 

## Requirements 
In order to run, the following Python modules       
are required:

- Numpy, SciPy, Sklearn, seaborn, statistics, Matplotlib
- alive_progress
- os, sys, argparse, glob
- PyTorch, torchvision
- gitpython
- ACSConv
- imblearn