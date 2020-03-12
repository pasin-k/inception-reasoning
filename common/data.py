import os
import random
import tensorflow as tf
import json
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
import pathlib
import numpy as np

def get_data_coco():  # Unused
    # Reference: https://www.tensorflow.org/tutorials/text/image_captioning?hl=zh-tw
    # Download caption annotation files
    annotation_folder = './data/annotations/'
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath('.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)
    else:
        annotation_file = os.path.abspath('.') + '/annotations/captions_train2014.json'

    # Download image files
    image_folder = './data/train2014/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('.') + image_folder

    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '{:012d}.jpg'.format(image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)

    # Select the first 30000 captions from the shuffled set
    num_examples = 100
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]


def download_cifar100():
    def save_data_and_label(folder_name):
        # Read data from binary
        with open(os.path.join(data_dir, folder_name), 'rb') as f:
            f.seek(0)
            data_dict = pickle.load(f, encoding='latin1')
        folder_name = folder_name + '_data'
        if not os.path.isdir(os.path.join(data_dir, folder_name)):
            os.makedirs(os.path.join(data_dir, folder_name))

        # Save as png image
        for i, name in enumerate(data_dict['filenames']):
            data = data_dict['data'][i, :].reshape((3, 32, 32))
            data = data.transpose(1, 2, 0).astype('uint8')
            plt.imsave(os.path.join(data_dir, folder_name, name), data, vmin=0, vmax=255)

        # Save label
        with open(os.path.join(data_dir, folder_name + '_label.json'), 'w') as filehandle:
            json.dump({'filenames': data_dict['filenames'], 'label': data_dict['fine_labels']}, filehandle)

    image_folder = './data/cifar-100-python/'
    # Load dataset if not found one
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                                            extract=True)
        data_dir = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)

        print("Decoding images")
        save_data_and_label('train')
        save_data_and_label('test')
    else:
        print("Use existing data")
        data_dir = os.path.abspath('.') + image_folder
    return data_dir


class Data():
    def __init__(self, data_folder=None, seed=None):
        if data_folder is None:
            self.data_folder = './data/cifar-100-python/'
        else:
            self.data_folder = data_folder
        random.seed(seed)

    def load_data(self, num_data = None):
        train_dir = os.path.join(self.data_folder, 'train_data')
        train_file_names = [os.path.abspath(f) for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
        with open(os.path.join(self.data_folder, 'train_data_label.json'), 'r') as f:
            train_file_label = json.load(f)['label']
        if num_data is not None:
            pairs = list(zip(train_file_names, train_file_label))
            pairs = random.choices(pairs, k=num_data)
            train_file_names, train_file_label = zip(*pairs)

        print(train_file_names)
        print(len(train_file_names))
        print(len(train_file_label))



if __name__ == '__main__':
    our_data = Data(seed=100)
    our_data.load_data(num_data=10)

# Unused
# def load_to_tfrecords(data_path):
#     data_dir = pathlib.Path(os.path.join(data_path, 'train_data'))
#     train_ds = tf.data.Dataset.list_files(str(data_dir/'*'))
#     for f in train_ds.take(100):
#         print(f.numpy())
#     # train_ds = tf.data.Dataset.list_files(os.path.join(data_path, 'train_data', '.png*'))
#     label_path = os.path.join(data_path, 'train_data_label.json')
#     train_ds = train_ds.map(lambda x: process_path(x, label_path))
#     for image, label in train_ds.take(2):
#         print("Image shape: ", image.numpy().shape)
#         print("Label: ", label.numpy())
#
#     test_ds = tf.data.Dataset.list_files(os.path.join(data_path, 'test_data', '*.png*'))
#
#
# def get_label(file_path):
#     with open(file_path) as filehandle:
#         labels = json.load(filehandle)
#     return tf.convert_to_tensor(labels['label'],dtype=tf.int32)
#
#
# def decode_img(img, width, height):
#     img = tf.image.decode_png(img, channels=3)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     return tf.image.resize(img, [width, height])
#
#
# def process_path(data_path, label_path):
#     label = get_label(label_path)
#     # load the raw data from the file as a string
#     img = tf.io.read_file(data_path)
#     img = decode_img(img, width=299, height=299)
#     print(label)
#     return img, label
