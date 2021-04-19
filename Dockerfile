FROM python:3.9

WORKDIR /tmp/ft_linear_regression
COPY *.py ./
COPY requirements.txt ./
COPY resources ./resources

ENV DISPLAY=host.docker.internal:0

RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install -y zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

CMD [ "zsh" ]
