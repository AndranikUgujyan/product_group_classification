from product_model import logger
from product_model.data_proc.normalizer import TextNormalizer
from product_model.saving_loading.model_saving_loading import ModelSavingLoading
import tensorflow as tf


class ModelPredictor:

    def __init__(self, model_path: str):
        loader = ModelSavingLoading()
        self._model = loader.load_model(model_path)
        self._normalizer = TextNormalizer()

    def predict(self, product_text):
        logger.debug('start prediction')
        pred_prob = self._model.predict([product_text])
        label_dict = {0: 'BICYCLES', 1: 'CONTACT LENSES', 2: 'USB MEMORY', 3: 'WASHINGMACHINES'}
        prediction_result = label_dict[pred_prob[0]]
        return prediction_result
