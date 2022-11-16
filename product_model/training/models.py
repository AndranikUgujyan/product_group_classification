import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras import layers
import product_model
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow_addons.metrics import FBetaScore, F1Score
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from keras.callbacks import EarlyStopping
from product_model import app_config

from keras.preprocessing.text import Tokenizer

ABS_DIR_PATH = os.path.dirname(os.path.abspath(product_model.__file__))
SAVE_DIR_PATH = os.path.join(ABS_DIR_PATH, app_config["logs_save_dir_path"])


class ModelNB:
    def model(self, train_sent, train_lab):
        model = Pipeline([("tfidf", TfidfVectorizer(min_df=0.,
                                                    max_df=1.,
                                                    norm='l2',
                                                    use_idf=True,
                                                    smooth_idf=True)), ("clf", MultinomialNB())])
        model.fit(train_sent, train_lab)
        return model


class ModelLR:
    def model(self, train_sent, train_lab):
        model = Pipeline([("tfidf", TfidfVectorizer(min_df=0.,
                                                    max_df=1.,
                                                    norm='l2',
                                                    use_idf=True,
                                                    smooth_idf=True)), ("clf", LogisticRegression())])
        model.fit(train_sent, train_lab)
        return model


class ModelKN:
    def model(self, train_sent, train_lab):
        model = Pipeline([("tfidf", TfidfVectorizer(min_df=0.,
                                                    max_df=1.,
                                                    norm='l2',
                                                    use_idf=True,
                                                    smooth_idf=True)), ("clf", KNeighborsClassifier())])
        model.fit(train_sent, train_lab)
        return model


class ModelRF:
    def model(self, train_sent, train_lab):
        model = Pipeline([("tfidf", TfidfVectorizer(min_df=0.,
                                                    max_df=1.,
                                                    norm='l2',
                                                    use_idf=True,
                                                    smooth_idf=True)), ("clf", RandomForestClassifier())])
        model.fit(train_sent, train_lab)
        return model
