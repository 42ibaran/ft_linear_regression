# ft_linear_regression

42 Project: ML algorithm to predict car price based on its milage.

## Usage
Project is divided into 2 parts: training and prediction programs.
### Training
To run training program, use this:
```
python train.py resources/data.csv
```
Program uses linear regression with gradient descent. After the execution a file `training_data.pk` with coefficient, constant, mean and standard deviation is created. It is used to retrieve coefficients for prediction program later on.

### Prediction
To run prediction program, use this:
```
python predict.py
```
Program will load the result of the training, promp you for milage input and give a prediction of the price. 

## Note
I used python 3.9, you can use whatever you want but if it crashes it's not my fault 😉 