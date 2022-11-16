import pickle

import tensorflow as tf
import tensorflow_addons as tfa
from product_model import logger
from product_model.utils.help_func import error_response


class ModelSavingLoading:

    def __init__(self):
        super(ModelSavingLoading, self).__init__()

    def load_model(self, model_path):
        logger.debug(f'Start loading model from {model_path}')
        try:
            with open(model_path, 'rb') as handle:
                model = pickle.load(handle)

            logger.debug(f'Loading complete from {model_path}')
            return model
        except Exception as err:
            return error_response(f'error={err}', logger)

    def save_model(self, model_for_save, model_path):
        logger.debug(f'Start saving model {model_for_save}')
        try:
            model_for_save.save(model_path)
            return True
        except Exception as err:
            return error_response(f'error={err}', logger)


# if __name__ == "__main__":
#     model_7_path = "/home/andranik/Desktop/imbd_classification_task/product_model/models/embedded_model_7"
#
#     ms = ModelSavingLoading()
#     model_1 = ms.load_model(model_7_path)
#
#
#     text = "HOLLANDRAD DAMEN 28 ZOLL TUSSAUD 3-GAENGE RH 5...	" \
#            "FAHRRAEDER // SPORTFAHRRAEDER	SCHALOW & KROH GMBH"
#
#     from product_model.data_proc.embedding import TextEmbedding
#
#     text = TextEmbedding().sent_embedding(text)
#     print(text)
#     pred_label = model_1.predict([text])
#     print(pred_label)
#     label_dict = {0: 'BICYCLES', 1: 'CONTACT LENSES', 2: 'USB MEMORY', 3: 'WASHINGMACHINES'}
#
#     print(label_dict[pred_label[0]])
