import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import os
import sys
import argparse


class DataPrepare:
    """
    DataPrepare will load the dataset, and convert to tfrecord
    """
    
    def __init__(self, data_name, destination):
        self.data_name = data_name
        self.destination = Path(destination)        
        assert self.data_name in ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']
        
        if self.data_name == 'mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        elif self.data_name == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        elif self.data_name == 'cifar100':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()
        elif self.data_name == 'fashion_mnist':
            (self.x_train ,self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            logging.error(f'The data_name should be in ["mnist", "cifar10", "cifar100" or "fashion_mnist"], not {self.data_name}')

        # Expand Dim for channel
        if len(self.x_train.shape) != 4:
            self.x_train = np.expand_dims(self.x_train, -1)
            self.x_test = np.expand_dims(self.x_test, -1)
        
    
    def _bytes_list(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        bytes_list = tf.train.BytesList(value=[value])
        return tf.train.Feature(bytes_list=bytes_list)
    
    def _int64_list(self, value):
        int64_list = tf.train.Int64List(value=[value])
        return tf.train.Feature(int64_list=int64_list)
    
    def _float_list(self, value):
        float_list = tf.train.FloatList(value=[value])
        return tf.train.Feature(float_list=float_list)
    
    def create_proto_msg(self, feat0, feat1, feat2, feat3, feat4):
        feature = {'image_height':self._int64_list(feat0),
                   'image_width': self._int64_list(feat1),
                   'image_channel':self._int64_list(feat2),
                   'image':self._bytes_list(feat3),
                   'label':self._int64_list(feat4)}
        
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        return example.SerializeToString()
    
    def _create_record(self, x, y, channel):
        
        with tf.io.TFRecordWriter(path = str(self.destination/channel) + '.tfrecord') as writer:
            for n, (img, label) in enumerate(zip(x, y)):
                img_height = img.shape[0]
                img_width = img.shape[1]
                img_channel = img.shape[2]
                img_feature = np.reshape(img, img_height * img_width * img_channel).tobytes()
                example = self.create_proto_msg(img_height, img_width, img_channel, img_feature, label)
                if n % 1000 == 0:
                    print(f'{channel} -- Processing Write TFRecord {n}')
                writer.write(example)
    
    def create_dataset(self):
        # Create Train dataset
        self._create_record(self.x_train, self.y_train, 'train')
        self._create_record(self.x_test, self.y_test, 'test')
        
                       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--destination', type=str, default='./data')
    args = parser.parse_args()
    if os.path.exists(args.destination):
        print(f"There are DESTINATION :: {args.destination}, will overwrite the data")
    else:
        os.makedirs(args.destination, exist_ok=True)
    data_creator = DataPrepare(args.data_name, args.destination)
    data_creator.create_dataset()
    
    