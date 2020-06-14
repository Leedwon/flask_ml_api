from joblib import load

# todo make singleton
class ModelProvider:
    def __init__(self):
        pass

    def get_model(self):
        model = load('model.pkl')
        return model


model = ModelProvider().get_model()
