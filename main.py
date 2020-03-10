from model import InceptionV3
import os, sys
import linecache
import numpy as np
from lime import lime_image
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import time


def get_imagenet_to_label():
    imagenet_code_to_label = {}
    with open(".\imagenet_to_label.txt") as f:
        lines = f.readlines()

    for line in lines:
        temp = line.replace('{', '').replace('}', '').split(':')
        imagenet_code_to_label[int(temp[0])] = temp[1].replace('\'', '').strip()
    return imagenet_code_to_label


if __name__ == '__main__':
    model = inc_net.InceptionV3()
    model.summary()


    img = image.load_img('./data/bed2.jpg', target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.vstack([x])

    prediction = model.predict(x)

    for i in decode_predictions(prediction):
        print(i)

    start = time.time()
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    print(explanation)

    x = 0
    decoder = get_imagenet_to_label()
    print(explanation.top_labels[x], decoder[explanation.top_labels[x]])

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[x], positive_only=False, num_features=10,
                                                hide_rest=False)
    print("Explanation time", time.time() - start)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()
