from tensorflow.keras.applications import ResNet50


# todo make singleton
class ModelProvider:
    def __init__(self):
        pass
        self._model = ResNet50(weights='imagenet')

    def get_model(self):
        return self._model
