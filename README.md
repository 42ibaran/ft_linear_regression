# ft_linear_regression

42 Project: Machine learning algorithm to predict car price based on its mileage. The model uses linear regression with gradient descent. 

## Setup
### Locally
You can run the project locally. Make sure you have python installed. I used v3.9, other versions might work too but who knows. To install dependencies run: 
```
pip3 install -r requirements.txt
```

### With Docker
You can also use the project with Docker. First, build the image:
```
docker build -t ft_linear_regression .
```
Then, you can run the container like so:
```
docker run -it --rm ft_linear_regression
```
In the container, all dependencies are installed. However, you might have problems with plotting the data due to display forwarding. To make it work (for Mac, not Windows, for Ubuntu see below) make sure that XQuartz is running and connection from remote clients is allowed:

<p align="center">
  <img src="https://raw.githubusercontent.com/42ibaran/ft_linear_regression/master/readme_img/xquartz_setting.png">
</p>

Then on the host run:
```
xhost + 127.0.0.1
```
to allow window forwarding from localhost. That should do it.

On Ubuntu what might work is if you run:
```
xhost + local:root
docker run -it --rm --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" ft_linear_regression 
```
Not sure about other Linux systems. Sorry 😢

### With VSCode
I kept `.devcontainer` directory with a setup for development using VSCode Remote Development extension. You can reopen the project directory using the extension, similarly to using Docker container but with more functionality.

## Usage
To run training program, run:
```
python train.py [-p] resources/data.csv
```
After the execution, a file `training_data.pk` with the result of the training is created. It is used to retrieve coefficients for the prediction program later on.


To run prediction program, use this:
```
python predict.py
```
It will load the result of the training, prompt you for mileage input and give a prediction of the price. 
