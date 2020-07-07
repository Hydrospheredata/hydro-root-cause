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

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

COPY --chown=app:app requirements.txt requirements.txt
RUN pip3 install --user -r requirements.txt

WORKDIR /app

COPY --chown=app:app anchor2/ /app/anchor2
COPY --chown=app:app rise/ /app/rise

RUN pip3 install --user anchor2/ rise/

COPY --chown=app:app . /app
RUN printf '{"version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json && \
    rm -rf .git

EXPOSE 5000
