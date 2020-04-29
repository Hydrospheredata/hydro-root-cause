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
RUN printf '{"version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json && \
    rm -rf .git

COPY . /app
EXPOSE 5000