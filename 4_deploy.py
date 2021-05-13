import os
import subprocess

if __name__ =="__main__":
    os.environ['MODEL_DIR'] = 'model'
    
    os.system('tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=fashion_model \
  --model_base_path="basic/deploymodel/version"')

    # or Using Docker 
    # ALL path required absolute path
    # DOCKER COMMAND
    
    # docker run -p 8501:8501 --mount type=bind,source="/home/han/GitHub/aws_mlops_sample/basic/deploymodel/version",target="/models/mnist" -e MODEL_NAME=mnist -t tensorflow/serving
    
    