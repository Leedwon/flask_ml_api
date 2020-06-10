import io

import flask
from model_provider import ModelProvider
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import numpy as np

app = flask.Flask(__name__)
model_provider = ModelProvider()
model = model_provider.get_model()


# returns numpy array
def prepare_image(image: Image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


@app.route('/predict', methods=['POST'])
def predict():
    result = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, (224, 224))

            predictions = model.predict(image)
            results = imagenet_utils.decode_predictions(predictions)
            result["predictions"] = []

            for (id, label, probability) in results[0]:
                res = {"label": label, "probability": float(probability)}
                result["predictions"].append(res)
        result["success"] = True

    return flask.jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
