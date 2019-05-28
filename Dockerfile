FROM python:3.6.8-slim-stretch

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential \
                       python3-pip \
                       python3-numpy \
                       python3-scipy \
                       libatlas-dev \
                       libatlas3-base

RUN  update-alternatives --set libblas.so.3 \
    /usr/lib/atlas-base/atlas/libblas.so.3 && \
    update-alternatives --set liblapack.so.3 \
    /usr/lib/atlas-base/atlas/liblapack.so.3

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

#RUN apt-get install -y python-matplotlib

#RUN mkdir  /var/log/celery


WORKDIR /app

COPY anchor2/ /app/anchor2
COPY rise/ /app/rise
RUN pip3 install anchor2/ rise/

COPY . /app
EXPOSE 5000

#COPY celery.service /etc/systemd/system/celery.service
#COPY celery.conf /etc/conf.d/celery
#
#RUN  apt-get install -y --reinstall systemd
#RUN  systemctl daemon-reload

