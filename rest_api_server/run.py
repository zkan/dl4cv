import io

import cv2
import flask
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
from PIL import Image


app = flask.Flask(__name__)
model = None
lb = None
classes = None


def prepare_image(image):
    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = img_to_array(image, data_format='channels_first') / 255.0
    image = np.expand_dims(image, axis=0)

    return image


@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'success': False
    }
    if flask.request.method == 'POST':
        image = flask.request.files.get('image').read()
        image = prepare_image(image)

        results = model.predict(image)

        i = results.argmax(axis=1)[0]
        label = lb.classes_[i]

        data['prediction'] = label
        data['class_probabilites'] = {
            label: str(prob) for label, prob in zip(classes, results[0])
        }
        data['success'] = True

    return flask.jsonify(data)


if __name__ == '__main__':
    model = load_model('../output/cnn.h5')

    f = open('../output/label-bins.pkl', 'rb')
    lb = pickle.load(f)
    f.close()

    classes = lb.classes_

    app.run(debug=False, threaded=False)
