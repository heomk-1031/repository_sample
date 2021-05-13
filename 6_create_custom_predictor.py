import tensorflow as tf
import tarfile
import os
import argparse


class TFModel(tf.Module):
    def __init__(self, model, img_size):
        self.model = model
        self.img_size = img_size
        
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
    def predict(self, img):
        """
        argument is img
        so the serving format should be 
        {'inputs':{'img':data.tolist()}}
        """
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize(img, (self.img_size, self.img_size))
        img = tf.cast(img, tf.float32) / 255.
        
        pred = self.model(img)
        return {'Prediction':pred}
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./deploymodel')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--model_path', type=str, default='./model/model_trained.tar.gz')
    args = parser.parse_args()
    
    print('Load Original Model')
    with tarfile.open(args.model_path, 'r') as model_tar:
        model_tar.extractall('./model_load')
    model = tf.keras.models.load_model('./model_load')
    

    version = args.version
    
    model_creator = TFModel(model=model, img_size=args.img_size)
    
    print('Saving Serving with Preprocessing Model')
    
    tf.keras.models.save_model(model_creator.model, filepath=os.path.join(args.save_dir, str(version)), 
                               save_format='tf',
                               signatures={'serving_default':model_creator.predict})
