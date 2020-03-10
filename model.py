import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pickle
from common import data
from sklearn.model_selection import train_test_split


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

        # Load Model
        if self.model_dir is None:
            self.model = self._get_pretrained_model()
            self.model_dir = './model/inceptionv3_model'
        else:
            try:
                self.model = tf.keras.models.load_model(model_dir)
            except:
                raise NameError("Model not found")
        self.model.summary()

        # Select dataset folder
        if self.data_dir is None:
            # Load cifar100 dataset
            self.data_dir = data.download_cifar100()

        # Load label name from meta file
        with open(os.path.join(self.data_dir, 'meta'), 'rb') as f:
            f.seek(0)
            meta = pickle.load(f, encoding='latin1')
            self.label_name = meta['fine_label_names']  # ['fine_label_names', 'coarse_label_names']

        # Load data
        data.load_to_tfrecords(self.data_dir)



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

if __name__ == '__main"__':
    my_model = InceptionV3()
    # my_model.model.summary()
    print(my_model.label_name)
