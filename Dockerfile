FROM python:3.6.8-slim-stretch

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y build-essential python3-numpy python3-scipy \
                     libatlas-dev libatlas3-base

RUN  update-alternatives --set libblas.so.3 \
    /usr/lib/atlas-base/atlas/libblas.so.3 && \
    update-alternatives --set liblapack.so.3 \
    /usr/lib/atlas-base/atlas/liblapack.so.3

RUN apt-get install -y python-matplotlib python3-pip supervisor

RUN mkdir  /var/log/celery

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY . /app

COPY supervisor.conf /etc/supervisor.conf

RUN pip3 install anchor2/
RUN pip3 install rise/


EXPOSE 5000


CMD supervisord -c /etc/supervisor.conf && python app.py
