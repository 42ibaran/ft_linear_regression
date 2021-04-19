import linear_regression as lr
from custom_errors import ModelNotTrainedError
from logger import * 

model = lr.Model(lr.WITH_TRAINING_DATA)

mileage = input("Provide mileage: ")

try:
    mileage = int(mileage)
except ValueError:
    log.error("Invalid mileage value.")
    exit(1)

try:
    prediction = model.predict(mileage)
    print(int(prediction))
except ValueError:
    log.error("Unable to predict, try training again.")
    exit(1)
