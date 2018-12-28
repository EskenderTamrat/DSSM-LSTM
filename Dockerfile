FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3.6 \
        python3-pip \
        python3.6-dev \
        libmysqlclient-dev \
        python3-setuptools \
        curl \
        cmake \
        wget \
        docutils-common

COPY requirements.txt /tmp

WORKDIR /tmp

RUN curl "https://bootstrap.pypa.io/get-pip.py" | python3.6
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN python3.6 -m pip install -r requirements.txt

COPY . /DSSM-LSTM

WORKDIR /DSSM-LSTM

#EXPOSE jsonrpc port
EXPOSE 8001

RUN python3.6 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service_spec/LSTM.proto

CMD [ "python3.6" , "server.py" ]