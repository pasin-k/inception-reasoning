import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

tfds.disable_progress_bar()


class InceptionV3():
    def __init__(self, model_dir=None, data_dir=None):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.IMG_SIZE = 299
        self.BATCH_SIZE = 32
        self.SHUFFLE_BUFFER_SIZE = 1000
        self.learning_rate = 0.0001
        self.model = None

        if self.model_dir is None:
            self.model = self._get_pretrained_model()
            self.model_dir = './model/inceptionv3_model'
        else:
            try:
                self.model = tf.keras.models.load_model(model_dir)
            except:
                raise NameError("Model not found")
        self.model.summary()

        if self.data_dir is None:
            self.download_cifar100()

        with open(os.path.join(self.data_dir, 'meta'), 'rb') as f:
            f.seek(0)
            meta = pickle.load(f, encoding='latin1')
            self.label_name = meta['fine_label_names']  # ['fine_label_names', 'coarse_label_names']



    @staticmethod
    def _get_pretrained_model():
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        base_model.trainable = False
        pool_layer = tf.keras.layers.GlobalAveragePooling2D()
        out_layer = tf.keras.layers.Dense(100)
        model = tf.keras.Sequential([base_model, pool_layer, out_layer])
        return model

    def train(self):
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    def get_data_coco(self):  # Unused
        # Reference: https://www.tensorflow.org/tutorials/text/image_captioning?hl=zh-tw
        # Download caption annotation files
        annotation_folder = '/annotations/'
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
        image_folder = '/train2014/'
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
        self.train_captions = train_captions[:num_examples]
        self.img_name_vector = img_name_vector[:num_examples]

    def download_cifar100(self):
        image_folder = '/cifar-100-python/'
        if not os.path.exists(os.path.abspath('.') + image_folder):
            image_zip = tf.keras.utils.get_file('train2014.zip',
                                                cache_subdir=os.path.abspath('.'),
                                                origin='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                                                extract=True)
            self.data_dir = os.path.dirname(image_zip) + image_folder
            os.remove(image_zip)
        else:
            print("Use existing data")
            self.data_dir = os.path.abspath('.') + image_folder

        def save_data_and_label(folder_name):
            with open(os.path.join(self.data_dir, folder_name), 'rb') as f:
                f.seek(0)
                dict = pickle.load(f, encoding='latin1')
            folder_name = folder_name + '_data'
            if not os.path.isdir(os.path.join(self.data_dir, folder_name)):
                os.makedirs(os.path.join(self.data_dir, folder_name))
            for i, name in enumerate(dict['filenames']):
                data = dict['data'][i, :].reshape((3,32,32))
                data = data.transpose(1,2,0).astype('uint8')
                plt.imsave(os.path.join(self.data_dir, folder_name, name), data, vmin=0, vmax=255)

        save_data_and_label('train')
        save_data_and_label('test')

        # image_folder = '/train2014/'
        # if not os.path.exists(os.path.abspath('.') + image_folder):
        #
        # (raw_train, raw_validation, raw_test), metadata = tfds.load('cifar100', split=['train[:80%]', 'train[80%:90%]',
        #                                                                                'train[90%:]'], with_info=True,
        #                                                             as_supervised=True)
        # print(raw_train)
        # print(metadata)
        #
        # def format_data(image, label):
        #     image = tf.cast(image, tf.float32)
        #     image = (image / 127.5) - 1
        #     image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        #     return image, label
        #
        #
        # train = raw_train.map(format_data)
        # validation = raw_validation.map(format_data)
        # test = raw_test.map(format_data)
        # train_batches = train.shuffle(self.SHUFFLE_BUFFER_SIZE).batch(self.BATCH_SIZE)
        # validation_batches = validation.batch(self.BATCH_SIZE)
        # test_batches = test.batch(self.BATCH_SIZE)
        #
        # for image_batch, label_batch in train_batches.take(1):
        #     pass
        # print(image_batch.shape)
        # print(label_batch)
        # print(self.model(image_batch).shape)


my_model = InceptionV3()
# my_model.model.summary()
print(my_model.label_name)
