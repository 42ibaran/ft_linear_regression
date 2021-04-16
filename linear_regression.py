import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from custom_errors import EmptyDataError, InvalidFlagError
from logger import * 

### TODO
# error management

TRAIN_ITER = 1000
LEARNING_RATE = 0.01
TRAINING_RESULT_FILENAME = "training_data.pk"

WITH_TRAINING_DATA = 1
EMPTY = 2

class Model():

    x = None
    y = None
    m = 0
    b = 0
    mean = 0
    std = 1

    # Y = m * X + b
    def __init__(self, flag=EMPTY):
        if flag == EMPTY:
            return
        elif flag == WITH_TRAINING_DATA:
            try:
                with open(TRAINING_RESULT_FILENAME, "rb") as fi:
                    self.m, self.b, self.mean, self.std = pickle.load(fi)
            except FileNotFoundError:
                log.error("Training data file (%s) not found." % TRAINING_RESULT_FILENAME)
                exit(1)
        else:
            raise InvalidFlagError("Invalid initialization flag provided.")

    def set_training_data(self, x, y):
        self.x = x
        self.y = y

        if len(self.x) == 0 or len(self.y) == 0:
            raise EmptyDataError("Data you provided is empty. \x46\x75\x63\x6b\x20\x79\x6f\x75\x2e")

    def feature_scale_normalize(self):
        if self.x is None or self.y is None:
            raise EmptyDataError("Can't normalize empty data.")

        self.mean = self.x.mean()
        self.std = self.x.std()
        self.x = (self.x - self.mean) / self.std

    def train(self):
        if self.x is None or self.y is None:
            raise EmptyDataError("Can't train on empty data.")

        for _ in range(TRAIN_ITER):
            error = self.guess(self.x) - self.y
            self.m -= (LEARNING_RATE / len(self.x)) * np.sum(error * self.x)
            self.b -= (LEARNING_RATE / len(self.x)) * error.sum()

    def guess(self, x):
        return self.m * x + self.b

    def plot(self):
        if self.x is None or self.y is None:
            raise EmptyDataError("Can't plot empty data.")

        _, ax = plt.subplots()
        ax.plot(self.x, self.y, 'b.')

        xSpaces = np.linspace(self.x.min(), self.x.max(), 100)
        ax.plot(xSpaces, self.m * xSpaces + self.b, 'b-')
        plt.show()

    def save(self):
        with open(TRAINING_RESULT_FILENAME, "wb") as fi:
            pickle.dump((self.m, self.b, self.mean, self.std), fi)

    def predict(self, x):
        xNorm = (x - self.mean) / self.std
        return self.guess(xNorm)
