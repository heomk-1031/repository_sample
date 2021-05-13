import tensorflow as tf
import numpy as np
import json
import requests


# Extract sample data
if __name__ == "__main__":
    print("""Model is Deployed via \n docker run -p 8501:8501 --mount type=bind,source="/home/han/GitHub/aws_mlops_sample/basic/deploymodel/version",target="/models/mnist" -e MODEL_NAME=mnist -t tensorflow/serving""")

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = np.expand_dims(x_test, -1)
    x_test.shape

    sample = x_test[0]

    data = sample.tolist()

    template = {'inputs':{'img':data}}

    payload = json.dumps(template)


    json_file = open('test_prediction.json', 'w')
    json_file.write(payload)
    json_file.close()


    headers = {"content-type": "application/json"}
    
    response = requests.post('http://localhost:8501/v1/models/mnist:predict',
                             data=payload,
                             headers=headers)

    response

    responsed = str(response.content.decode())
    print(responsed)