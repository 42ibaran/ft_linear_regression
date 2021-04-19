import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from custom_errors import InvalidDataError, InvalidFlagError
from logger import * 

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
    error = []

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
            raise InvalidDataError("Data you provided is empty. \x46\x75\x63\x6b\x20\x79\x6f\x75\x2e")
        elif np.isnan(x).any() or np.isnan(y).any():
            raise InvalidDataError("Data you provided sucks.")

    def value_normalize(self, x):
        if self.std == 0:
            raise ValueError("Standard deviation is 0. Check your training data.")
        return (x - self.mean) / self.std

    def feature_scale_normalize(self):
        if self.x is None or self.y is None:
            raise InvalidDataError("Can't normalize empty data.")

        self.mean = self.x.mean()
        self.std = self.x.std()
        self.x = self.value_normalize(self.x)

    def train(self):
        if self.x is None or self.y is None:
            raise InvalidDataError("Can't train on empty data.")

        for _ in range(TRAIN_ITER):
            error = self.guess(self.x) - self.y
            self.m -= (LEARNING_RATE / len(self.x)) * np.sum(error * self.x)
            self.b -= (LEARNING_RATE / len(self.x)) * error.sum()

            self.error.append(-error.sum() / len(self.x))

    def guess(self, x):
        return self.m * x + self.b

    def plot(self):
        if self.x is None or self.y is None:
            raise InvalidDataError("Can't plot empty data.")

        xDenorm = self.x * self.std + self.mean

        _, ax = plt.subplots()
        ax.plot(xDenorm, self.y, 'b.')

        xSpaces = np.linspace(xDenorm.min(), xDenorm.max(), 2)
        ySpaces = (self.m / self.std) * (xSpaces - self.mean) + self.b
        ax.plot(xSpaces, ySpaces, 'r-')
        ax.legend(['Dataset points', 'Prediction function'])
        ax.set_xlabel('Mileage, km')
        ax.set_ylabel('Price')
        plt.grid()
        plt.show()

    def save(self):
        with open(TRAINING_RESULT_FILENAME, "wb") as fi:
            pickle.dump((self.m, self.b, self.mean, self.std), fi)
            log.info("Training result saved successfully.")

    def predict(self, x):
        x = self.value_normalize(x)
        return self.guess(x)
