import os
import random
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input as prepro_inp
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


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label):
    image_shape = tf.image.decode_png(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def download_cifar100():
    """
    Download cifar100 dataset, convert data into tfrecords for pipeline process during data fetching
    @return: Create directory "./data/cifar-100-python" containing tfrecords and original images
    """

    def _save_data_and_label(save_dir):
        # Read data from binary
        with open(save_dir, 'rb') as f:
            f.seek(0)
            data_dict = pickle.load(f, encoding='latin1')
        data_dir = save_dir + '_data'
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        # Save binary to png image
        for i, name in enumerate(data_dict['filenames']):
            data = data_dict['data'][i, :].reshape((3, 32, 32))
            data = data.transpose(1, 2, 0).astype('uint8')
            plt.imsave(os.path.join(data_dir, name), data, vmin=0, vmax=255)

        # Save label as json file
        with open(os.path.join(data_dir + '_label.json'), 'w') as filehandle:
            json.dump({'filenames': data_dict['filenames'], 'label': data_dict['fine_labels']}, filehandle)

        # Collect image_path and label and serialize to tfrecord
        image_path_list = [os.path.join(data_dir, name) for name in data_dict['filenames']]
        label_list = data_dict['fine_labels']

        record_file = save_dir + '.tfrecords'
        with tf.io.TFRecordWriter(record_file) as writer:
            for filename, label in zip(image_path_list, label_list):
                image_string = open(filename, 'rb').read()
                tf_example = image_example(image_string, label)
                writer.write(tf_example.SerializeToString())

    image_folder = './data/cifar-100-python/'
    # Load dataset if not found one
    if not os.path.exists(image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('./data'),
                                            origin='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                                            extract=True)
        # data_dir = os.path.join(os.path.dirname(image_zip),image_folder)
        os.remove(image_zip)

        print("Decoding images and save to .tfrecords")
        _save_data_and_label(os.path.join(image_folder, 'train'))
        _save_data_and_label(os.path.join(image_folder, 'test'))
        os.remove(os.path.join(image_folder, 'train'))
        os.remove(os.path.join(image_folder, 'test'))
        os.remove(os.path.join(image_folder, 'meta'))
    else:
        print("Use existed data instead")
    return image_folder


class Dataset():
    def __init__(self, data_folder_path=None, seed=None, image_size=(299, 299), batch_size=8):
        if data_folder_path is None:
            self.data_folder_path = './data/cifar-100-python/'
        else:
            self.data_folder_path = data_folder_path
        random.seed(seed)
        self.image_size = image_size
        self.data_index = None
        self.train_data = None
        self.test_data = None
        self.BATCH_SIZE = batch_size

    def get_images(self, image_dir, num_data):
        """
        Get image from given directory
        @param image_dir: string of folder directory where images are stored. All file in folder must only be images
        @param num_data: Number of data randomly sampled. If None, will load all data
        @return: Array of images of shape (num_data, data_size, data_size)
        """
        # Fetch image paths
        image_file_names = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                            os.path.isfile(os.path.join(image_dir, f))]
        image_file_names.sort()
        print(image_file_names[0:100])
        # Randomly select some if num_data is not None
        if num_data is not None:
            if self.data_index is None:
                self.data_index = random.sample(range(len(image_file_names)), num_data)
            image_file_names = [image_file_names[i] for i in self.data_index]

        # Load images
        image_data = []
        for img_path in image_file_names:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.image_size)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = prepro_inp(x)
            image_data.append(x)
        return np.vstack(image_data)

    def get_labels(self, label_path, num_data):
        """
        Get label data from a json file. Json file should have a dictionary 'filenames' and 'label' with list of
        all label in it. In this case, we sort the label by index of filenames
        @param label_path: String of directory to json file
        @param num_data:
        @return: Array of images of shape (num_data, data_size, data_size)
        """
        with open(label_path, 'r') as f:
            data = json.load(f)
            label_data = data['label']
            label_name = data['filenames']

        label_data = [l for _, l in sorted(zip(label_name, label_data))]
        # label_name = [l for l,_ in sorted(zip(label_name,label_data))]

        # Randomly select some if num_data is not None
        if num_data is not None:
            if self.data_index is None:
                self.data_index = random.sample(range(len(label_data)), num_data)
            label_data = [label_data[i] for i in self.data_index]
        return np.array(label_data)

    def load_data_tfrecord(self):
        def _deserialize(example):
            # Create a dictionary describing the features.
            image_feature_description = {
                'height': tf.io.FixedLenFeature([], tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64),
                'depth': tf.io.FixedLenFeature([], tf.int64),
                'label': tf.io.FixedLenFeature([], tf.int64),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
            }
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example, image_feature_description)

        def _decode(data_dict):
            img = tf.image.decode_png(data_dict['image_raw'], channels=3)
            img = tf.image.resize(img, self.image_size)
            img = prepro_inp(img)
            label = data_dict['label']
            return img, label

        raw_train_dataset = tf.data.TFRecordDataset(os.path.join(self.data_folder_path, "train.tfrecords"))
        train_dataset = raw_train_dataset.map(_deserialize)
        train_dataset = train_dataset.map(_decode)
        train_dataset = train_dataset.shuffle(500).batch(self.BATCH_SIZE)
        # train_dataset = train_dataset.repeat(None)
        self.train_data = train_dataset.prefetch(buffer_size=None)

        raw_test_dataset = tf.data.TFRecordDataset(os.path.join(self.data_folder_path, "test.tfrecords"))
        test_dataset = raw_test_dataset.map(_deserialize)
        test_dataset = test_dataset.map(_decode)
        self.test_data = test_dataset.batch(self.BATCH_SIZE)
        return self.train_data, self.test_data

    def _load_image_and_label(self, image_dir, label_dir, num_data):
        return {"image": self.get_images(image_dir, num_data),
                "label": self.get_labels(label_dir, num_data)}

    def load_data_numpy(self, train_num_data=None, test_num_data=None, data_type=None):
        """
        Load images as numpy and label from data_folder_path
        @param train_num_data: Number of data (train)to load. If None, will fetch all data in the folder
        @param test_num_data: Number of data (test) to load. If None, will fetch all data in the folder
        @param data_type: String, if data_type is 'tf', will return as tf.dataset
        @return: Return numpy array of data and label
        """

        train_image_dir = os.path.join(self.data_folder_path, 'train_data')
        train_label_dir = os.path.join(self.data_folder_path, 'train_data_label.json')
        train_data = self._load_image_and_label(train_image_dir, train_label_dir, train_num_data)

        self.data_index = None
        test_image_dir = os.path.join(self.data_folder_path, 'test_data')
        test_label_dir = os.path.join(self.data_folder_path, 'test_data_label.json')
        test_data = self._load_image_and_label(test_image_dir, test_label_dir, test_num_data)

        if data_type is None:
            self.train_data = train_data
            self.test_data = test_data
        elif data_type.lower() == 'tf':  # Convert to tf.Dataset
            self.train_data = tf.data.Dataset.from_tensor_slices((train_data['image'], train_data['label']))
            self.test_data = tf.data.Dataset.from_tensor_slices((test_data['image'], test_data['label']))
        else:
            raise KeyError("Daty type invalid")
        return self.train_data, self.test_data

    def preprocess_input(self, inp):
        return prepro_inp(inp)

if __name__ == '__main__':
    our_data = Dataset()
    train_data, test_data = our_data.load_data_numpy(train_num_data=100, test_num_data=100)
    # print(y_train)
