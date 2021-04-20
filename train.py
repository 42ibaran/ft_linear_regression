import argparse
import numpy as np
import pandas as pd

import linear_regression as lr
from custom_errors import InvalidDataError
from logger import * 

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("-p", help="plot data after training", action='store_true')
args = parser.parse_args()

try:
    df = pd.read_csv(args.filename)
except:
    log.error("Data file (%s) not found or invalid csv file." % args.filename)
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
except (InvalidDataError, ValueError) as e:
    log.error(e)
    exit(1)

model.train()
model.save()

if args.p:
    model.plot()
