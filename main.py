from model import InceptionV3
from common.data import Dataset
import os, sys
import linecache
import urllib.request

from lime import lime_image
from tensorflow.keras.applications import inception_v3 as inc_net

from tensorflow.keras.preprocessing import image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import time


def get_imagenet_to_label():
    imagenet_code_to_label = {}
    with open("./imagenet_to_label.txt") as f:
        lines = f.readlines()

    for line in lines:
        temp = line.replace('{', '').replace('}', '').split(':')
        imagenet_code_to_label[int(temp[0])] = temp[1].replace('\'', '').strip()
    return imagenet_code_to_label


#Referencing lime: https://github.com/marcotcr/lime
if __name__ == '__main__':
    my_model = InceptionV3()
    my_model.model.summary()
    # my_model = inc_net.InceptionV3()

    data = Dataset()

    data.load_data_tfrecord()
    print(data.train_data)
    # my_model.train(data, epochs=2)

    if not os.path.isdir("./data/test/"):
        os.makedirs("./data/test/")
    urllib.request.urlretrieve("https://farm4.static.flickr.com/3282/2581806386_7b042493b5.jpg",
                               "./data/test/car_01.jpg")

    img = image.load_img("./data/test/car_01.jpg", target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = data.preprocess_input(x)
    x = np.vstack([x])

    prediction = my_model.predict(x)

    for i in my_model.decode_predict(prediction):
        print(i)

    start = time.time()
    explainer = lime_image.LimeImageExplainer(verbose=True)
    explanation = explainer.explain_instance(x[0], my_model.predict, top_labels=5, hide_color=0, num_samples=100000)
    print(explanation)

    x = 1
    decoder = get_imagenet_to_label()
    print(explanation.top_labels[x], decoder[explanation.top_labels[x]])

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[x], positive_only=False, num_features=3,
                                                hide_rest=False)

    print("Explanation time", time.time() - start)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()
