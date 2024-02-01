import sys
import argparse
from random import *

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from src import tensorfi_plus as tfi_batch
from src.utility import get_fault_injection_configs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='vgg16',
                        help='Name of DNN model')
    parser.add_argument('--layer_no', default=0, type=int,
                        help='Layer to inject faults')
    args = parser.parse_args()
    model_name = args.model_name
    layer_no = args.layer_no
    print("Random layer for fault injection " + str(layer_no))
    file_object = open('cifar100_logs/cifar100_final_log_' + model_name + '_' + str(layer_no) + '_layer.txt', 'a')
    inputs = np.load("cifar100_test_inputs.npy")
    labels = np.load("cifar100_test_labels.npy")

    K.clear_session()
    model = tf.keras.models.load_model('cifar100_keras_pretrained/' + model_name + '.h5')

    data_count, _, _, _ = inputs.shape
    yaml_file = "confFiles/sample1.yaml"
    model_graph, super_nodes = get_fault_injection_configs(model)
    total_injection = 1000
    for i in range(total_injection):
        rand_num = randint(0, data_count - 1)
        img = tf.expand_dims(inputs[rand_num], axis=0)
        predicted_label = model.predict(img).argmax(axis=-1)[0]
        res = tfi_batch.inject(model=model, x_test=img, confFile=yaml_file,
                               model_graph=model_graph, super_nodes=super_nodes, layer_inject_index=layer_no)
        faulty_prediction = res.final_label[0]
        # print(str(data_partition * 2000 + i) + " : " + str(y_val[data_partition * 2000 + i]) + " : " + str(predicted_label) + " : " + str(faulty_prediction))
        file_object.write(str(i) + " : " + str(rand_num) + " : " + str(labels[rand_num]) + " : " + str(predicted_label) + " : " + str(faulty_prediction))
        file_object.write("\n")
        file_object.flush()
    file_object.close()
    print("Done")


if __name__ == '__main__':
    main()
