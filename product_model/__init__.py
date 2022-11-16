import logging.config
from logging import getLogger
from product_model.utils.help_func import load_cfg

app_config = load_cfg('configs/app.yml')
log_file = load_cfg('configs/logging.yml')
logging.config.dictConfig(log_file)
logger = getLogger('product_model')

