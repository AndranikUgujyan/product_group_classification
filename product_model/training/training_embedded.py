import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import plotly.express as px  # for data visualization
import plotly.graph_objects as go  # for data visualization
from xgboost import XGBClassifier
import product_model
from product_model import app_config
from product_model.training.evaluation import ModelPrediction
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from product_model.training.evaluation import ModelPrediction

RANDOM_STATE = 42

abs_dir_path = os.path.dirname(os.path.abspath(product_model.__file__))

ALL_MODELS_RESULT_PLOT_ABS_PATH = os.path.join(abs_dir_path, app_config["all_models_result_plot_path"])

train_data_abs_embedded_path = os.path.join(abs_dir_path, app_config['train_embedded_data_path'])
test_data_abs_embedded_path = os.path.join(abs_dir_path, app_config['test_embedded_data_path'])

train_df = pd.read_csv(train_data_abs_embedded_path)
test_df = pd.read_csv(test_data_abs_embedded_path)

train_df.text_embedded = train_df.text_embedded.apply(lambda x: [float(y) for y in x[1:-1].split()])
test_df.text_embedded = test_df.text_embedded.apply(lambda x: [float(y) for y in x[1:-1].split()])

train_X = list(train_df["text_embedded"])
train_y = list(train_df["product_group_cat"])

test_X = list(test_df["text_embedded"])
test_y = list(test_df["product_group_cat"])

model_rfc = RandomForestClassifier()
model_svc = SVC()
model_lg = LogisticRegression()
model_dtc = DecisionTreeClassifier()
model_knc = KNeighborsClassifier()
model_gbc = GradientBoostingClassifier()
rfc_model_fit = model_rfc.fit(train_X, train_y)
svc_model_fit = model_svc.fit(train_X, train_y)
lg_model_fit = model_lg.fit(train_X, train_y)
dtc_model_fit = model_dtc.fit(train_X, train_y)
knc_model_fit = model_knc.fit(train_X, train_y)
gbc_model_fit = model_gbc.fit(train_X, train_y)



mp = ModelPrediction(test_X, test_y)

model_rfc_results = mp.pred(model_rfc, "Random Forest Classifier Embedded")
model_svc_results = mp.pred(model_svc, "SVC Embedded")
model_lg_results = mp.pred(model_lg, "Logistic Regression Embedded")
model_knc_results = mp.pred(model_knc, "K Neighbors Classifier Embedded")
model_dtc_results = mp.pred(model_dtc, "Decision Tree Classifier Embedded")
model_gbc_results = mp.pred(model_gbc, "Gradient Boosting Classifier Embedded")

all_model_results = pd.DataFrame({
    "baseline_lr": model_lg_results,
    "baseline_rf": model_rfc_results,
    "baseline_kn": model_knc_results,
    "baseline_svc": model_svc_results,
    "baseline_dtc": model_dtc_results,
    "baseline_gbc": model_gbc_results})
all_model_results = all_model_results.transpose()
print(all_model_results)

model_6_abs_path = os.path.join(abs_dir_path, app_config['model_6_path'])
model_7_abs_path = os.path.join(abs_dir_path, app_config['model_7_path'])
model_8_abs_path = os.path.join(abs_dir_path, app_config['model_8_path'])
model_9_abs_path = os.path.join(abs_dir_path, app_config['model_9_path'])
model_10_abs_path = os.path.join(abs_dir_path, app_config['model_10_path'])
model_11_abs_path = os.path.join(abs_dir_path, app_config['model_11_path'])

pickle.dump(model_rfc, open(model_6_abs_path, 'wb'))
pickle.dump(model_svc, open(model_7_abs_path, 'wb'))
pickle.dump(model_lg, open(model_8_abs_path, 'wb'))
pickle.dump(model_knc, open(model_9_abs_path, 'wb'))
pickle.dump(model_dtc, open(model_10_abs_path, 'wb'))
pickle.dump(model_gbc, open(model_11_abs_path, 'wb'))


