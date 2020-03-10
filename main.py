from model import InceptionV3
import os,sys
import linecache
import numpy as np
from lime import lime_image
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from skimage.segmentation import mark_boundaries

if __name__ == '__main__':
    model = inc_net.InceptionV3()

    explainer = lime_image.LimeImageExplainer()


    img = image.load_img('./data/automobile1.png', target_size=(299,299))
    print(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    x = np.vstack([x])

    prediction = model.predict(x)

    for i in decode_predictions(prediction):
        print(i)

    explanation = explainer.explain_instance(x[0], model.predict, top_labels=5,num_samples=1000)
    print(explanation)
