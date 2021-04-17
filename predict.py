import linear_regression as lr
from custom_errors import ModelNotTrainedError
from logger import * 

model = lr.Model(lr.WITH_TRAINING_DATA)

milage = input("Provide milage: ")

try:
    milage = int(milage)
except ValueError:
    log.error("Invalid milage value.")
    exit(1)

print(int(model.predict(milage)))
