import tensorflow as tf
import numpy as np
import tarfile
import os
import logging
import argparse
import json

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
    
    with tarfile.open(args.model_path, 'r') as model_tar:
        model_tar.extractall('./model_load')
        
    with tarfile.open(args.test_data_path, 'r') as test_data_tar:
        test_data_tar.extractall('.')
    
     
    print('Load Test Dataset')
    test_data = tf.data.experimental.load('./prep_test/', element_spec=(tf.TensorSpec(shape=(None, args.img_size, args.img_size, args.img_channels), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(None, ), dtype=tf.int64)))
    test_data = test_data.map(lambda img, label: resize_img(img, label, args.img_size))
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)
    
    
    print('Load Model')
    model = tf.keras.models.load_model('./model_load')
    

    print(f"Evaluating Model Name: {args.model_path.split('.')[0]}")
    result_metrics = tf.keras.metrics.Mean(name='evaluation')
    result_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    model.evaluate(test_data)
    for n, (img, label) in enumerate(test_data):
        pred = model(tf.cast(img, tf.float32))
        metrics_result = result_metrics(result_accuracy(label, pred))

    final_evaluation = result_metrics.result()
    result_json_template = {'EvaluationResult':f'{final_evaluation}'}
    
    print(result_json_template)
    with open(args.result_path, 'w') as report:
        json.dump(result_json_template, report) 
    
    