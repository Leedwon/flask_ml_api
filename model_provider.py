import tensorflow


class ModelProvider:
    def __init__(self):
        pass

    def get_model(self):
        model = tensorflow.keras.models.load_model('model')
        return model
