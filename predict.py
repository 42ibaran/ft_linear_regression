import linearRegression as lr
from customErrors import ModelNotTrainedError

try:
    model = lr.Model(lr.WITH_TRAINING_DATA)
except FileNotFoundError:
    print("File with training data doesn't exist")
    exit(1)

milage = float(input("Provide milage: "))
print(int(model.predict(milage)))
