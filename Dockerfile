FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        python3.6 \
        python3.6-dev \
        python3-pip \
        python3-setuptools

COPY requirements.txt /tmp

WORKDIR /tmp

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN python3.6 -m pip install -r requirements.txt

COPY . /DSSM-LSTM

WORKDIR /DSSM-LSTM

#EXPOSE jsonrpc port
EXPOSE 8001
EXPOSE 6205

RUN python3.6 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service/LSTM.proto

CMD [ "python3.6" , "service/server.py" ]
