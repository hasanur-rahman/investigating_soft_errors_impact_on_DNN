import os
from random import *
import re
import sys
from glob import glob
from pathlib import Path

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg16, vgg19, resnet, xception, nasnet, mobilenet, mobilenet_v2, \
    inception_resnet_v2, inception_v3, densenet

from src import tensorfi_plus as tfi_batch
from src.utility import get_fault_injection_configs


def get_model_from_name(model_name):
    if model_name == "ResNet50":
        return resnet.ResNet50()
    elif model_name == "ResNet101":
        return resnet.ResNet101()
    elif model_name == "ResNet152":
        return resnet.ResNet152()
    elif model_name == "VGG16":
        return vgg16.VGG16()
    elif model_name == "VGG19":
        return vgg19.VGG19()
    elif model_name == "Xception":
        return xception.Xception()
    elif model_name == "NASNetMobile":
        return nasnet.NASNetMobile()
    elif model_name == "NASNetLarge":
        return nasnet.NASNetLarge()
    elif model_name == "MobileNet":
        return mobilenet.MobileNet()
    elif model_name == "MobileNetV2":
        return mobilenet_v2.MobileNetV2()
    elif model_name == "InceptionResNetV2":
        return inception_resnet_v2.InceptionResNetV2()
    elif model_name == "InceptionV3":
        return inception_v3.InceptionV3()
    elif model_name == "DenseNet121":
        return densenet.DenseNet121()
    elif model_name == "DenseNet169":
        return densenet.DenseNet169()
    elif model_name == "DenseNet201":
        return densenet.DenseNet201()


def get_preprocessed_input_by_model_name(model_name, x_val):
    if model_name == "ResNet50" or model_name == "ResNet101" or model_name == "ResNet152":
        return resnet.preprocess_input(x_val)
    elif model_name == "VGG16":
        return vgg16.preprocess_input(x_val)
    elif model_name == "VGG19":
        return vgg19.preprocess_input(x_val)
    elif model_name == "Xception":
        return xception.preprocess_input(x_val)
    elif model_name == "NASNetMobile" or model_name == "NASNetLarge":
        return nasnet.preprocess_input(x_val)
    elif model_name == "MobileNet":
        return mobilenet.preprocess_input(x_val)
    elif model_name == "MobileNetV2":
        return mobilenet_v2.preprocess_input(x_val)
    elif model_name == "InceptionResNetV2":
        return inception_resnet_v2.preprocess_input(x_val)
    elif model_name == "InceptionV3":
        return inception_v3.preprocess_input(x_val)
    elif model_name == "DenseNet121" or model_name == "DenseNet169" or model_name == "DenseNet201":
        return densenet.preprocess_input(x_val)


def get_data_path_by_model_name(model_name, path_imagenet_val_dataset):
    if model_name == "ResNet50" or model_name == "ResNet101" or model_name == "ResNet152" or model_name == "VGG16" \
            or model_name == "VGG19" or model_name == "NASNetMobile" or model_name == "MobileNet" \
            or model_name == "MobileNetV2" or model_name == "DenseNet121" or model_name == "DenseNet169" \
            or model_name == "DenseNet201" or model_name == "EfficientNetB0":
        return str(path_imagenet_val_dataset) + "/sampled_x_val_224_"
    elif model_name == "Xception" or model_name == "InceptionResNetV2" or model_name == "InceptionV3":
        return str(path_imagenet_val_dataset) + "/sampled_new_x_val_299_1.npy"
    elif model_name == "NASNetLarge":
        return str(path_imagenet_val_dataset) + "/sampled_new_x_val_331_1.npy"


def main():
    # model_name = sys.argv[1]
    # process_no = int(sys.argv[2])
    model_names = ["ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19", "Xception", "NASNetMobile", "NASNetLarge", "MobileNet", "MobileNetV2", "InceptionResNetV2", "InceptionV3", "DenseNet121", "DenseNet169", "DenseNet201"]
    for model_name in model_names:
        model = get_model_from_name(model_name)
        print("Model name : " + model_name)
        print("Total layers " + str(len(model.layers)))


if __name__ == '__main__':
    main()
