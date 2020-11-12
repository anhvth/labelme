FROM ubuntu:bionic
LABEL maintainer "Kentaro Wada <www.kentaro.wada@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  apt-get update -qq && \
  apt-get install -qq -y \
    git 

# install miniconda
RUN apt-get update

RUN apt-get install -y wget 
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN apt-get install -y     python3 \
    python3-pip \
    python3-matplotlib \
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN conda --version

RUN python -m pip install -U pip setuptools wheel

RUN pip install git+https://github.com/anhvth/pyson
RUN git clone https://github.com/anhvth/labelme.git /labelme
RUN pip install -e /labelme
RUN apt-get update && apt-get install vim -y
ENV LANG en-US

#ENTRYPOINT [ "labelme" ]
