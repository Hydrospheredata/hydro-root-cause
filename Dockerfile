FROM python:3.7.4-slim-stretch

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential \
                       python3-pip \
                       python3-numpy \
                       python3-scipy \
                       libatlas-dev \
                       libatlas3-base \
                       git

RUN  update-alternatives --set libblas.so.3 \
    /usr/lib/atlas-base/atlas/libblas.so.3 && \
    update-alternatives --set liblapack.so.3 \
    /usr/lib/atlas-base/atlas/liblapack.so.3

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY anchor2/ /app/anchor2
COPY rise/ /app/rise

RUN pip3 install anchor2/ rise/

COPY . /app
EXPOSE 5000


ENV HTTP_UI_ADDRESS http://managerui:80
ENV GRPC_UI_ADDRESS managerui:9090

ENV MONGO_URL mongodb
ENV MONGO_PORT 27017
ENV MONGO_AUTH_DB admin

ENV ROOT_CAUSE_DB_NAME root_cause

ENV AWS_ACCESS_KEY_ID minio
ENV AWS_SECRET_ACCESS_KEY minio123
ENV S3_ENDPOINT http://minio:9000

ENV DEBUG True