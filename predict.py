import linear_regression as lr
from custom_errors import ModelNotTrainedError
from logger import * 

model = lr.Model(lr.FROM_BINARY)

mileage = input("Provide mileage: ")

try:
    mileage = int(mileage)
except ValueError:
    log.error("Invalid mileage value.")
    exit(1)

if mileage < 0:
    log.error("Negative mileage, really?")
    exit(1)

try:
    prediction = model.predict(mileage)
    if prediction < 0:
        log.warning("Predicted price is negative, defaulting to 0.")
        prediction = 0
    print(int(prediction))
except ValueError:
    log.error("Unable to predict, try training again.")
    exit(1)
