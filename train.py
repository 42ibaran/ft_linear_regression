import argparse
import numpy as np
import pandas as pd

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

df = pd.read_csv(args.filename)

x = np.array(df.get('km'))
y = np.array(df.get('price'))

model = Model()

model.set_training_data(x, y)
model.feature_scale_normalize()
model.train()
model.save()
model.plot()
