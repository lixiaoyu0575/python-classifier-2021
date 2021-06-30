FROM anibali/pytorch:cuda-10.1
USER root
#RUN sudo su
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
#RUN chmod -R 777 /
## The MAINTAINER instruction sets the author field of the generated images.

MAINTAINER author@example.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
#RUN chown -R user:user /physionet
#USER user
#RUN mkdir /phsionet/model
#RUN apt-get update && apt-get upgrade -y && apt-get clean
#RUN apt-get install -y python3.6 python3.6-dev python3-pip
#RUN ln -s /usr/bin/python3.6 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt 
# -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
