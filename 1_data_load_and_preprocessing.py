import tensorflow as tf
import numpy as np
import json
import argparse
import os
import tarfile


class PreprocessData:
    def __init__(self, train_data_path, test_data_path, batch_size):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        
        self.feature_dict = {'image_height': tf.io.FixedLenFeature([], tf.int64),
                             'image_width': tf.io.FixedLenFeature([], tf.int64),
                             'image_channel': tf.io.FixedLenFeature([], tf.int64),
                             'image': tf.io.FixedLenFeature([], tf.string),
                             'label': tf.io.FixedLenFeature([], tf.int64)}

    def parse_data(self, data):
        data = tf.io.parse_single_example(data, features=self.feature_dict)
        return data
    
    def create_supervised(self, data):
        shape = (data['image_height'], data['image_width'], data['image_channel'])
        img = data['image']
        label = data['label']
        return (shape, img, label)
    
    def reshape_data(self, shape, img, label):
        h = shape[0]
        w = shape[1]
        c = shape[2]
        
        img = tf.io.decode_raw(img, out_type=tf.uint8)
        img = tf.reshape(img, shape=(h, w, c))
        return (img, label)
    
    def normalize_data(self, img, label):
        img = tf.cast(img, tf.float32) / 255.
        return (img, label)
    
    def load_data(self, data_path):
        data = tf.data.TFRecordDataset(data_path)
        data = data.map(self.parse_data)
        data = data.map(self.create_supervised)
        data = data.map(self.reshape_data)
        data = data.map(self.normalize_data)
        data = data.batch(self.batch_size)
        return data
        
    def get_supervised_data(self):
        train_data = self.load_data(self.train_data_path)
        test_data = self.load_data(self.test_data_path)
        return train_data, test_data
    
class WriteTFRecord:
    def __init__(self, destination):
        self.destination = destination
    
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
    
    
    def create_proto_msg(self, feat0, feat1):
        feature = {'image':self._bytes_list(feat0.numpy().tobytes()),
                   'label':self._bytes_list(feat1.numpy().tobytes())}
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        return example.SerializeToString()
    
    def tf_serialize_example(self, feat0, feat1):
        tf_string = tf.py_function(self.create_proto_msg, (feat0, feat1), tf.string)
        return tf.reshape(tf_string, ())

    def write_tfrecord(self, channel):
        dst_name = os.path.join(self.destination, f'prep_{channel}.tfrecord')
        # example = self.data.unbatch()
        example = self.data.map(self.tf_serialize_example)

        writer = tf.data.experimental.TFRecordWriter(dst_name)
        writer.write(example)
        print(f'File Write on {dst_name}')
    
    def save_data(self, data, channel):
        save_path = os.path.join(self.destination, 'prep_' + channel)
        print(f'Saving {channel} at {save_path}')
        tf.data.experimental.save(data, path=save_path)
        print('Complete.. Saving\n')
        return save_path
        
    def create_archive(self, path, channel):
        print(f'Start Achiving {path} -- {channel}')
        with tarfile.open(os.path.join(self.destination, 'prep_' + channel + '.tar.gz'), 'w:gz') as tar:
            tar.add(path, arcname=os.path.basename(path))
        print(f"Complete -- \n{os.path.join(self.destination, 'prep_' + channel + '.tar.gz')}\n")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/train.tfrecord')
    parser.add_argument('--test_data_path', type=str, default='./data/test.tfrecord')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--destination', type=str, default='./data/preprocess')
    args = parser.parse_args()
    
    if os.path.exists(args.destination):
        print(f"There are DESTINATION :: {args.destination}, will overwrite the data")
    else:
        os.makedirs(args.destination, exist_ok=True)
    
    data_loader = PreprocessData(args.train_data_path, args.test_data_path, args.batch_size)
    train_data, test_data = data_loader.get_supervised_data()
    
    
    data_writer = WriteTFRecord(args.destination)
    train_prep_path = data_writer.save_data(train_data, 'train')
    test_prep_path = data_writer.save_data(test_data, 'test')
    
    data_writer.create_archive(train_prep_path, 'train')
    data_writer.create_archive(test_prep_path, 'test')
    