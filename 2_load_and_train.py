import tensorflow as tf
import numpy as np
from simple_architecture import SimpleDense, SimpleConv
from tqdm import tqdm, trange
import os
import argparse
import sys
import logging
import tarfile
import pandas as pd


# TODO requires to modify to data preparation step
@tf.function
def resize_img(img, label, size):
    img = tf.image.resize(img, size=(size, size))
    return (img, label)
 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_tar_path', type=str, default='./data/preprocess')
    parser.add_argument('--test_tar_path', type=str, default='./data/preprocess')
    parser.add_argument('--model_save_path', type=str, default='./model/')
    parser.add_argument('--metric_report_path', type=str, default='.')
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--img_channels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_type', type=str, default='conv')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--units', type=str, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_activation', type=str, default='relu')
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--output_activation', type=str, default='softmax')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    
    args = parser.parse_args()    
    
    print('Extract Train Data')
    with tarfile.open(args.train_tar_path + '/prep_train.tar.gz', 'r') as train_tar:
        train_tar.extractall('.')
    with tarfile.open(args.test_tar_path + '/prep_test.tar.gz', 'r') as test_tar:
        test_tar.extractall('.')
    
    print('Loading Data')
    train_data = tf.data.experimental.load('./prep_train', element_spec=(tf.TensorSpec(shape=(None, args.img_size, args.img_size, args.img_channels), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(None, ), dtype=tf.int64)))
    
    test_data = tf.data.experimental.load('./prep_test/', element_spec=(tf.TensorSpec(shape=(None, args.img_size, args.img_size, args.img_channels), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(None, ), dtype=tf.int64)))
    
    print(f'Modify Data to fit the shape [{args.img_size},{args.img_size},{args.img_channels}]')
    train_data = train_data.map(lambda img, label: resize_img(img, label, args.img_size))
    train_data = train_data.shuffle(buffer_size=10000)
    train_data = train_data.unbatch()
    train_data = train_data.batch(args.batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    
    test_data = test_data.map(lambda img, label: resize_img(img, label, args.img_size))
    test_data = test_data.unbatch()
    test_data = test_data.batch(args.batch_size)
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)
    print('Modified Data\n', train_data)
    
    print(f'\n\nPrepare {args.model_type} for Training...\n')
    if args.model_type == 'conv':
        model = SimpleConv(args.units, args.num_layers, args.kernel_size, args.hidden_activation, args.output_dim, args.output_activation, args.dropout_rate)
    
    elif args.model_type == 'dense':
        model = SimpleDense(args.units, args.num_layers, args.hidden_activation, args.output_dim, args.output_activation)
    
    print('Start Training')
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])
    
    # save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.model_save_path, monitor='val_loss', verbose=1)
    
    hist = model.fit(train_data, epochs=args.epochs, validation_data=test_data) 
    tf.saved_model.save(model, args.model_save_path + '/')
    
    print('Complete Training')
    model_metrics = pd.DataFrame(hist.history)
    model_metrics.to_csv(os.path.join(args.metric_report_path, 'metrics.csv'))
    
    print('Archiving Artifact')
    with tarfile.open(os.path.join(args.model_save_path, 'model_trained.tar.gz'), 'w:gz') as tar:
        tar.add(args.model_save_path, arcname=os.path.basename(args.model_save_path))
    
    