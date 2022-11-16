import os
import numpy as np
import pandas as pd
import product_model
from matplotlib import pyplot as plt
import dataframe_image as dfi
from product_model import app_config
from product_model.training.evaluation import ModelPrediction
from product_model.training.models import ModelNB, ModelLR, ModelRF, ModelKN
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from product_model.utils.help_func import calculate_results
from product_model.training.nn_models import Model_LSTM, TokenizeText
import pickle

abs_dir_path = os.path.dirname(os.path.abspath(product_model.__file__))

ALL_MODELS_RESULT_PLOT_ABS_PATH = os.path.join(abs_dir_path, app_config["all_models_result_plot_path"])

train_data_abs_path = os.path.join(abs_dir_path, app_config['train_norm_data_path'])
test_data_abs_path = os.path.join(abs_dir_path, app_config['test_norm_data_path'])

train_df = pd.read_csv(train_data_abs_path)
test_df = pd.read_csv(test_data_abs_path)

train_sentences = train_df["text"].to_numpy()
train_labels = train_df["product_group_cat"].to_numpy()

test_sentences = test_df["text"].to_numpy()
test_labels = test_df["product_group_cat"].to_numpy()

model_nb = ModelNB().model(train_sentences, train_labels)
model_lr = ModelLR().model(train_sentences, train_labels)
model_rf = ModelRF().model(train_sentences, train_labels)
model_kn = ModelKN().model(train_sentences, train_labels)

MAX_SEQUENCE_LENGTH = 250
tok_text = TokenizeText(train_df["text"].values).tt()

tokenizer_path_abs_path = os.path.join(abs_dir_path, app_config['tokenizer_path'])
with open(tokenizer_path_abs_path, 'wb') as handle:
    pickle.dump(tok_text, handle, protocol=pickle.HIGHEST_PROTOCOL)
train_sentences_x = tok_text.texts_to_sequences(train_df["text"].values)
train_sentences_x = pad_sequences(train_sentences_x, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of train data tensor:', train_sentences_x.shape)
print(train_sentences_x.shape[1])
test_sentences_x = tok_text.texts_to_sequences(test_df["text"].values)
test_sentences_x = pad_sequences(test_sentences_x, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test data tensor:', test_sentences_x.shape)

train_labels_y = pd.get_dummies(train_df["product_group_cat"]).values
print('Shape of train label tensor:', train_labels_y.shape)

test_labels_y = pd.get_dummies(test_df['product_group_cat']).values
print('Shape of test label tensor:', test_labels_y.shape)

model_lstm = Model_LSTM(train_sentences_x, train_labels_y, train_sentences_x.shape[1]).model()

mp = ModelPrediction(test_sentences, test_labels)

model_nb_results = mp.pred(model_nb, "naive_bayes")
model_lr_results = mp.pred(model_nb, "logistic_regression")
model_rf_results = mp.pred(model_nb, "rf_classifier")
model_kn_results = mp.pred(model_nb, "kn_classifier")
model_lstm_results = ModelPrediction(test_sentences_x, test_labels_y).pred(model_lstm, "lstm")

all_model_results = pd.DataFrame({"baseline_nb": model_nb_results,
                                  "baseline_lr": model_lr_results,
                                  "baseline_rf": model_rf_results,
                                  "baseline_kn": model_kn_results,
                                  "baseline_lstm": model_lstm_results})
all_model_results = all_model_results.transpose()
print(all_model_results)

all_model_results["accuracy"] = all_model_results["accuracy"] / 100

all_models_result_plot_path = ALL_MODELS_RESULT_PLOT_ABS_PATH.format("all_models_norm_sampled.png")
all_models_result = all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig(all_models_result_plot_path, dpi=300)

all_models_results_f1_score_path = ALL_MODELS_RESULT_PLOT_ABS_PATH.format("all_models_f1_score_under_sampled.png")
all_model_results_fig = all_model_results.sort_values("f1", ascending=False)["f1"].plot(kind="bar", figsize=(10, 7))
plt.savefig(all_models_results_f1_score_path, dpi=300)

# # get the prediction probabilities from baseline model
# baseline_pred_probs = np.max(model_0.predict_proba(val_sentences), axis=1)
# model_2_pred_probs = tf.squeeze(tf_model_2.predict(val_sentences), axis=1)
# model_6_pred_probs = tf.squeeze(tf_model_6.predict(val_sentences))
# combined_pred_probs = baseline_pred_probs + model_2_pred_probs + model_6_pred_probs
# combined_preds = tf.round(combined_pred_probs / 3)
# print(combined_preds[:20])
# ensemble_results = calculate_results(val_labels, combined_preds)
# print(ensemble_results)
# all_model_results.loc["ensemble_results"] = ensemble_results
# all_model_results.loc["ensemble_results"]["accuracy"] = all_model_results.loc["ensemble_results"]["accuracy"] / 100
# print(all_model_results)
# df_styled = all_model_results.style.background_gradient()
#
# all_models_results_path = ALL_MODELS_RESULT_PLOT_ABS_PATH.format("all_models_f1_score_df_under_sampled.png")
# dfi.export(df_styled, all_models_results_path)
#
model_1_abs_path = os.path.join(abs_dir_path, app_config['model_1_path'])
model_2_abs_path = os.path.join(abs_dir_path, app_config['model_2_path'])
model_3_abs_path = os.path.join(abs_dir_path, app_config['model_3_path'])
model_4_abs_path = os.path.join(abs_dir_path, app_config['model_4_path'])
model_5_abs_path = os.path.join(abs_dir_path, app_config['model_5_path'])
# model_6_abs_path = os.path.join(abs_dir_path, app_config['model_6_path'])
# model_7_abs_path = os.path.join(abs_dir_path, app_config['model_7_path'])
#



pickle.dump(model_nb, open(model_1_abs_path, 'wb'))
pickle.dump(model_lr, open(model_2_abs_path, 'wb'))
pickle.dump(model_rf, open(model_3_abs_path, 'wb'))
pickle.dump(model_kn, open(model_4_abs_path, 'wb'))
model_lstm.save(model_5_abs_path)
