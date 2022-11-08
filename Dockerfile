FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y wget

#conda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Create the environment:
COPY requirements.txt /code/requirements.txt
RUN conda create --name myenv python=3.8
#do these first since they are also in frozen pip list and will not be found in listed channels
RUN conda install --name myenv pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
RUN conda install --name myenv -c rapidsai -c nvidia -c conda-forge -c defaults cuml=0.18 python=3.8 cudatoolkit=11.0

RUN /bin/bash -c "conda run -n myenv pip install -r /code/requirements.txt"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python3", "/code/app.py"]


