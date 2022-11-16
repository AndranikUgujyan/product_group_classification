import os
import json
import os
import product_model
from flask import Flask, request
from product_model.data_proc.embedding import TextEmbedding
from product_model import app_config, logger
from product_model.predicter.model_predictor import ModelPredictor
from product_model.utils.help_func import error_response, get_input_text

ABS_DIR_PATH = os.path.dirname(os.path.abspath(product_model.__file__))

app = Flask(__name__)


@app.route('/identify_product', methods=['POST'])
def predict_product():
    try:
        log_headers_and_ip(request)
        _data = request.data.decode('utf-8')
        logger.debug(f'started, data:{_data}')
        j_obj = json.loads(_data)
        logger.debug('get client instance')
        model_path = config_model(j_obj['data'][0]["model"])

        model_client = ModelPredictor(model_path)
        texts = get_input_text(j_obj['data'][0], logger)
        print("This is text", texts)
        if "embedded" in model_path:
            texts = TextEmbedding().sent_embedding(texts)

        prediction = model_client.predict(texts)
        response = app.response_class(response=json.dumps([prediction], indent=True),
                                      status=200,
                                      mimetype='application/json')
        return response
    except Exception as err:
        return error_response(f'error={err}', logger)


def config_model(model_name):
    try:
        if model_name == "naive_bayes":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_1_path'])
            return model_path
        if model_name == "lr":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_2_path'])
            return model_path
        if model_name == "rf":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_3_path'])
            return model_path
        if model_name == "kn":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_4_path'])
            return model_path
        if model_name == "conv1d":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_5_path'])
            return model_path
        if model_name == "rfc_emb":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_6_path'])
            return model_path
        if model_name == "svc_emb":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_7_path'])
            return model_path
        if model_name == "lg_emb":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_8_path'])
            return model_path
        if model_name == "knc_emb":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_9_path'])
            return model_path
        if model_name == "dtc_emb":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_10_path'])
            return model_path
        if model_name == "gbc_emb":
            model_path = os.path.join(ABS_DIR_PATH, app_config['model_11_path'])
            return model_path
    except Exception as err:
        return error_response(f'error={err}', logger)


def log_headers_and_ip(request):
    logger.debug('started')
    try:
        logger.debug(f'IP:{request.remote_addr}')
        logger.debug(f'headers:{request.headers}')
    except Exception:
        logger.exception('unable to log headers and IP.')


@app.errorhandler(500)
def server_error(e):
    logger.exception('error occurred during a request.')
    return f"An internal error occurred: <pre>{e}</pre>See logs for full stacktrace.", 500


if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 8080)))
