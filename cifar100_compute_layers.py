import tensorflow as tf


def main():
    # model_name = sys.argv[1]
    # process_no = int(sys.argv[2])
    model_names = ["googlenet", "inceptionv3", "inceptionv4", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vgg11", "vgg13", "vgg16", "vgg19", "xception"]
    for model_name in model_names:
        model = tf.keras.models.load_model('cifar100_keras_pretrained/' + model_name + '.h5')
        print("Model name : " + model_name)
        print("Total layers " + str(len(model.layers)))


if __name__ == '__main__':
    main()
