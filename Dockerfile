FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir training
RUN mkdir data
RUN mkdir models

COPY training/finetune.py training
COPY training/pretrain.py training
COPY training/utils.py training
COPY data/utils.py data
COPY models/transformer.py models
COPY models/cnn.py models

ENV PYTHONPATH="/app"