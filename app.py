from flask import Flask
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

app = Flask(__name__)

#load_model
print("loading Model")
json_file = open('model/model_pre/model.json', 'r')
model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(model_json)
model.load_weights("model/model_pre/model.h5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# def prepare_image(image):
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize((224, 224))
#     image = img_to_array(image)
#
#     image = image / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image

def prepare_image(image):
    image = img_to_array(image)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    image = image.reshape(1, 224, 224, 3)
    return image


@app.route("/")
def test():
    return "Working"


@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image)
        prediction = model.predict(image)
        return {"prediction": f"{prediction}"}


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

