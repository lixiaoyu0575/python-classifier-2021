FROM anibali/pytorch:cuda-10.1
USER root
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER author@example.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
RUN chmod -R 777 /physionet
USER user
#RUN apt-get update && apt-get upgrade -y && apt-get clean
#RUN apt-get install -y python3.6 python3.6-dev python3-pip
#RUN ln -s /usr/bin/python3.6 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
#RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
