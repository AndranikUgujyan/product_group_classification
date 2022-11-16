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


class Model_LSTM:
    def __init__(self,
                 train_sent,
                 train_labels,
                 input_length,
                 epochs_num=1,
                 batch_size=64,
                 max_nb_words=50000,
                 max_sequence_length=250,
                 embedding_dim=100):
        self.train_sent = train_sent
        self.train_labels = train_labels
        self.epochs_num = epochs_num
        self.batch_size = batch_size
        self.max_nb_words = max_nb_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.input_length = input_length

    def model(self):
        model = Sequential()
        model.add(Embedding(self.max_nb_words, self.embedding_dim, input_length=self.input_length))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        model.fit(self.train_sent,
                  self.train_labels,
                  epochs=self.epochs_num,
                  batch_size=self.batch_size,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        return model


class TokenizeText:
    def __init__(self,
                 train_sent,
                 max_nb_words=50000,
                 max_sequence_length=250):
        self.train_sent = train_sent
        self.max_nb_words = max_nb_words
        self.max_sequence_length = max_sequence_length

    def tt(self):
        tokenizer = Tokenizer(num_words=self.max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(self.train_sent)
        return tokenizer
