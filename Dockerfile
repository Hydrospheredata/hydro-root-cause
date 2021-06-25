FROM python:3.7.10-slim-stretch

RUN apt-get update && \
    apt-get install -y build-essential \
                       libatlas-dev \
                       libatlas3-base \
                       git curl

RUN  update-alternatives --set libblas.so.3 \
    /usr/lib/atlas-base/atlas/libblas.so.3 && \
    update-alternatives --set liblapack.so.3 \
    /usr/lib/atlas-base/atlas/liblapack.so.3

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

WORKDIR /app

COPY poetry.lock pyproject.toml ./
COPY anchor2/ ./anchor2
COPY rise/ ./rise
COPY anchor_tasks ./anchor_tasks
COPY rise_tasks ./rise_tasks
RUN ~/.poetry/bin/poetry config virtualenvs.create false
RUN ~/.poetry/bin/poetry install -n
COPY . .

RUN printf '{"version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json && \
    rm -rf .git


EXPOSE 5000

ENTRYPOINT ["bash", "/app/start.sh"]