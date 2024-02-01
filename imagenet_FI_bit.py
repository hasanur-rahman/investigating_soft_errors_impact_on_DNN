import os
import argparse
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='VGG16',
                        help='Name of DNN model')
    parser.add_argument('--bit_pos', default=0, type=int,
                        help='Layer to inject faults')
    args = parser.parse_args()
    model_name = args.model_name
    bit_pos = args.bit_pos
    print("Bit for fault injection " + str(bit_pos))
    file_object = open('imagenet_logs/imagenet_final_log_' + model_name + '_' + str(bit_pos) + '_bit.txt', 'a')
    for data_partition in range(5):
        path_imagenet_val_dataset = Path("imagenet_data/")  # path/to/data/
        y_val = np.load(str(path_imagenet_val_dataset / "y_val_sampled.npy"))
        x_val_path = get_data_path_by_model_name(model_name=model_name, path_imagenet_val_dataset=path_imagenet_val_dataset)

        K.clear_session()
        model = get_model_from_name(model_name)

        x_val = np.load(x_val_path + str(data_partition+1) + ".npy").astype('float32')
        x_val = get_preprocessed_input_by_model_name(model_name, x_val)
        data_count, _, _, _ = x_val.shape
        yaml_file = "confFiles/sample_bit_pos_" + str(bit_pos) + ".yaml"
        model_graph, super_nodes = get_fault_injection_configs(model)
        total_injection = 10
        for i in range(data_count):
            img = x_val[i]
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            predicted_label = model.predict(img).argmax(axis=-1)[0]
            for j in range(total_injection):
                res = tfi_batch.inject(model=model, x_test=img, confFile=yaml_file,
                                       model_graph=model_graph, super_nodes=super_nodes)
                faulty_prediction = res.final_label[0]
                # print(str(data_partition * 2000 + i) + " : " + str(y_val[data_partition * 2000 + i]) + " : " + str(predicted_label) + " : " + str(faulty_prediction))
                file_object.write(str(data_partition * 2000 + i) + " : " + str(y_val[data_partition * 2000 + i]) + " : " + str(predicted_label) + " : " + str(faulty_prediction))
                file_object.write("\n")
                file_object.flush()
    file_object.close()
    print("Done")


if __name__ == '__main__':
    main()
