import io
import json

import flask
from model_provider import ModelProvider
from PIL import Image
import numpy as np

from numpy_array_encoder import NumpyArrayEncoder

app = flask.Flask(__name__)
model_provider = ModelProvider()
model = model_provider.get_model()


# returns numpy array
def prepare_image(image: Image):
    img = np.resize(image, (28, 28))
    arr = np.array(img)
    arr = arr.reshape(1, 28, 28, 1)
    return arr


@app.route('/predict', methods=['POST'])
def predict():
    result = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image)).convert("L")

            image = prepare_image(image)

            predictions = model.predict(image)
            result["predictions"] = json.dumps(predictions, cls=NumpyArrayEncoder)
            predicted_class = np.argmax(predictions, axis=-1)
            result["predicted_class"] = json.dumps(predicted_class, cls=NumpyArrayEncoder)
        result["success"] = True

    return flask.jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
