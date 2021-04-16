import argparse
import numpy as np
import pandas as pd

import linear_regression as lr
from custom_errors import EmptyDataError
from logger import * 

log.basicConfig(format='%(levelname)s: %(message)s', level=log.WARN)

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

try:
    df = pd.read_csv(args.filename)
except FileNotFoundError:
    log.error("Data file (%s) not found." % args.filename)
    exit(1)

try:
    x = np.array(df.get('km'))
    y = np.array(df.get('price'))
except:
    log.error("Data file is invalid.")
    exit(1)

model = lr.Model()

try:
    model.set_training_data(x, y)
    model.feature_scale_normalize()
    model.train()
    model.save()
    model.plot()
except EmptyDataError as e:
    log.error(e)
    exit(1)
