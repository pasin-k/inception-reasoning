import os, sys
import linecache
import urllib.request
import numpy as np
import argparse
import time
import matplotlib
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from lime import lime_image
from tensorflow.keras.applications.inception_v3 import preprocess_input as prepro_inp
from tensorflow.keras.preprocessing import image

try:
    from .model import InceptionV3  # Use for django run
except ImportError:
    from model import InceptionV3  # Use for terminal run


# from .common.data import Dataset

def get_imagenet_to_label():
    imagenet_code_to_label = {}
    try:
        with open("./algorithm/imagenet_to_label.txt") as f:
            lines = f.readlines()
    except FileNotFoundError:
        with open("imagenet_to_label.txt") as f:
            lines = f.readlines()

    for line in lines:
        temp = line.replace('{', '').replace('}', '').split(':')
        imagenet_code_to_label[int(temp[0])] = temp[1].replace('\'', '').strip()
    return imagenet_code_to_label

def get_model(test_image_path):
    my_model = InceptionV3()
    img = image.load_img(test_image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = prepro_inp(x)
    x = np.vstack([x])

    return x[0], my_model


def predict(test_image_path):
    my_model = InceptionV3()
    my_model.model.summary()
    # my_model = inc_net.InceptionV3()

    # data = Dataset()
    # data.load_data_tfrecord()
    # print(data.train_data)
    # my_model.train(data, epochs=2)

    img = image.load_img(test_image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = prepro_inp(x)
    x = np.vstack([x])

    prediction = my_model.predict(x)

    # for i in my_model.decode_predict(prediction):
    #     print(i)
    return x, my_model.decode_predict(prediction), my_model


def explain(image, my_model, prediction_rank=0, show_img=True):
    start = time.time()
    explainer = lime_image.LimeImageExplainer(verbose=True)
    explanation = explainer.explain_instance(image, my_model.predict, top_labels=5, hide_color=0, num_samples=1000)
    # print(explanation)

    decoder = get_imagenet_to_label()
    print(explanation.top_labels[prediction_rank], decoder[explanation.top_labels[prediction_rank]])

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[prediction_rank], positive_only=False,
                                                num_features=5,
                                                hide_rest=False)  # num_features is top super pixel that gives positive value

    print("Explanation time", time.time() - start)
    if show_img:
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.show()
    masked_image = mark_boundaries(temp / 2 + 0.5, mask)
    return masked_image


# Referencing lime: https://github.com/marcotcr/lime
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=None)
    args = parser.parse_args()

    if args.image is None:
        if not os.path.isdir("./data/test/"):
            os.makedirs("./data/test/")
        urllib.request.urlretrieve("https://farm4.static.flickr.com/3282/2581806386_7b042493b5.jpg",
                                   "./data/test/car_01.jpg")
        test_image_path = "./data/test/car_01.jpg"
    else:
        test_image_path = args.image
    print("Using file from {}".format(test_image_path))
    images, prediction, my_model = predict(test_image_path)
    print("Decoding")
    print(prediction)
    explain(images[0], my_model, show_img=True)
