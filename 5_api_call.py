import tensorflow as tf
import numpy as np
import tarfile
import os
import logging
import argparse
import json
import requests

@tf.function
def resize_img(img, label, size):
    img = tf.image.resize(img, size=(size, size))
    return (img, label)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/model_trained.tar.gz')
    parser.add_argument('--test_data_path', type=str, default='./data/preprocess/prep_test.tar.gz')
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--img_channels', type=int, default=1)
    parser.add_argument('--result_path', type=str, default='./evaluation_report.json')
    
    args = parser.parse_args()
    
    with tarfile.open(args.test_data_path, 'r') as test_data_tar:
        test_data_tar.extractall('.')
        
    print('Load Test Dataset')
    test_data = tf.data.experimental.load('./prep_test/', element_spec=(tf.TensorSpec(shape=(None, args.img_size, args.img_size, args.img_channels), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(None, ), dtype=tf.int64)))
    test_data = test_data.map(lambda img, label: resize_img(img, label, args.img_size))
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)
    
    
    for sample in test_data.take(1):
        pass
    
    
    print('the data is already preprocessed')
    sample_img = sample[0].numpy()
    sample_img
    
    label = sample[1]
    
    data = json.dumps({'signature_name':'serving_default',
                       'instances': sample_img.tolist()})
    
    
    headers={'content-type':'application/json'}
    json_response = requests.post('http://localhost:8501/v1/models/test:predict', data=data, headers=headers)
    
    print(json_response)