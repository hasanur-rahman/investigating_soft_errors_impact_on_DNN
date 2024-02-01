# Investigating The Impact of Transient Hardware Faults on Deep Learning Neural Network Inference

This article is accepted in STVR'24. This journal is an extension to our PRDC'22 conference paper titled "Characterizing Deep Learning Neural Network Failures Between Algorithmic Inaccuracy and Transient Hardware Faults".

DOI of this article is [here](doi.ori/10.1002/stvr.1873).
Please cite our article from [here](https://onlinelibrary.wiley.com/action/showCitFormats?doi=10.1002%2Fstvr.1873&mobileUi=0).

## Installation

Prerequisites for installing the project,
1. Ubuntu OS (Tested with Ubuntu 18.04)
2. Python 3.6
3. Resolve all the dependency from requirements.txt file


## Run the following python scripts to reproduce the results:
```
cifar100_super_label_generate.py**: Sample code to generate super label mapping

sample_gtsrb_data.py: Sample 10000 images from 60000 test set of GTSRB dataset.
sample_imagenet_data.py: Sample 10000 images from 60000 test set.
imagenet_sampled_data_prepare.py: prepare sampled CIFAR100 data for inference.
imagenet_super_label_generate.py: generate super label for imagenet dataset
synset_to_keras_mapping: Prepare imagenet label

cifar100_final_FI.py : Store fault injection result for a particular range of Cifar100 images
parallel_processing_cifar100.py : Parallely execute fault injection on a specific model of cifar100 dataset. It uses cifar100_final_FI.py file.
merge_cifar100_result.py : Merge all the partial results generated from above command to a single file.
process_cifar100_result_av_mis_normalized.py : Compute SCM probability on initially fault free data.
process_cifar100_result_fault_free.py : Compute SCM probability of fault injected data.

gtsrb_final_FI.py : Store fault injection result for a particular range of gtsrb images
parallel_processing_gtsrb.py : Parallely execute fault injection on a specific model of gtsrb dataset. It uses gtsrb_final_FI.py file.
merge_gtsrb_result.py : Merge all the partial results generated from above command to a single file.
process_gtsrb_result_normalized.py: Compute SDC probability of GTSRB dataset

imagenet_final_FI.py : Store fault injection result for a particular range of imagenet images
parallel_processing_imagenet.py : Parallely execute fault injection on a specific model of imagenet dataset. It uses imagenet_final_FI.py file.
merge_imagenet_result.py : Merge all the partial results generated from above command to a single file.
process_imagenet_result_av_mis_normalized.py : Compute SCM probability on initially fault free data.
process_imagenet_result_fault_free.py : Compute SCM probability of fault injected data.
```
