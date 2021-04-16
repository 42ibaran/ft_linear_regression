import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from customErrors import *

### TODO
# Save thetas

TRAIN_ITER = 1000
LEARNING_RATE = 0.01

class Model():
    x = None
    y = None

    # Y = m * X + b
    def __init__(self, m = 0, b = 0):
        self.m = m
        self.b = b

    def set_training_data(self, x, y):
        self.x = x
        self.y = y

    def feature_scale_normalize(self):
        if self.x is None or self.y is None:
            raise EmptyDataError("Can't normalize empty data")

        self.mean = self.x.mean()
        self.std = self.x.std()
        self.x = (self.x - self.mean) / self.std

    def train(self):
        if self.x is None or self.y is None:
            raise EmptyDataError("Can't train on empty data")

        for _ in range(TRAIN_ITER):
            error = self.guess(self.x) - self.y
            self.m -= (LEARNING_RATE / len(self.x)) * np.sum(error * self.x)
            self.b -= (LEARNING_RATE / len(self.x)) * error.sum()
        return

    def guess(self, x):
        return self.m * x + self.b

    def plot(self):
        if self.x is None or self.y is None:
            raise EmptyDataError("Can't plot empty data")

        _, ax = plt.subplots()
        ax.plot(self.x, self.y, 'b.')

        xSpaces = np.linspace(self.x.min(), self.x.max(), 100)
        ax.plot(xSpaces, self.m * xSpaces + self.b, 'b-')
        plt.show()

    def save(self):
        return

    def predict(self, x):
        xNorm = (x - self.mean) / self.std
        return self.guess(xNorm)