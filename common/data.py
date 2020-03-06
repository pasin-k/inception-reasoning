import os
import tensorflow as tf
import json
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt

def get_data_coco():  # Unused
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
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]


def download_cifar100():
    def save_data_and_label(folder_name):
        with open(os.path.join(data_dir, folder_name), 'rb') as f:
            f.seek(0)
            dict = pickle.load(f, encoding='latin1')
        folder_name = folder_name + '_data'
        if not os.path.isdir(os.path.join(data_dir, folder_name)):
            os.makedirs(os.path.join(data_dir, folder_name))
        for i, name in enumerate(dict['filenames']):
            data = dict['data'][i, :].reshape((3 ,32 ,32))
            data = data.transpose(1 ,2 ,0).astype('uint8')
            plt.imsave(os.path.join(data_dir, folder_name, name), data, vmin=0, vmax=255)
        with open(os.path.join(data_dir, folder_name, 'label.json'), 'w') as filehandle:
            json.dump({'filenames': dict['filenames'], 'label' :dict['fine_labels']}, filehandle)

    image_folder = '/cifar-100-python/'
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