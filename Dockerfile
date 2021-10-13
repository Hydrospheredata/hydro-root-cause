# syntax=docker/dockerfile:1
FROM python:3.8.11-slim-bullseye as python-base
LABEL maintainer="support@hydrosphere.io"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_PATH=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.1.6 

ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgssapi-krb5-2>=1.18.3-6+deb11u1 \
    libk5crypto3>=1.18.3-6+deb11u1 \
    libkrb5-3>=1.18.3-6+deb11u1 \
    libkrb5support0>=1.18.3-6+deb11u1 \
    libssl1.1>=1.1.1k-1+deb11u1 \
    openssl>=1.1.1k-1+deb11u1 && \
    rm -rf /var/lib/apt/lists/*

FROM python-base AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl git && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python - && \
    mv /root/.poetry $POETRY_PATH && \
    python -m venv $VENV_PATH && \
    poetry config virtualenvs.create false && \
    rm -rf /var/lib/apt/lists/*

COPY poetry.lock pyproject.toml ./
COPY anchor2 ./anchor2
COPY anchor_tasks ./anchor_tasks
COPY rise_tasks ./rise_tasks
COPY rise ./rise
RUN poetry install --no-interaction --no-ansi -vvv

ARG GIT_HEAD_COMMIT
ARG GIT_CURRENT_BRANCH
COPY . ./
RUN if [ -z "$GIT_HEAD_COMMIT" ] ; then \
    printf '{"name": "hydro-root-cause", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json ; else \
    printf '{"name": "hydro-root-cause", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$GIT_HEAD_COMMIT" "$GIT_CURRENT_BRANCH" "$(python --version)" >> buildinfo.json ; \
    fi


FROM python-base as runtime

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

WORKDIR /app

COPY --from=build $VENV_PATH $VENV_PATH
COPY --from=build --chown=app:app start.sh logging_config.ini ./
COPY --from=build --chown=app:app rootcause ./rootcause
COPY --from=build --chown=app:app json_schemas ./json_schemas
COPY --from=build --chown=app:app anchor2 ./anchor2
COPY --from=build --chown=app:app anchor_tasks ./anchor_tasks
COPY --from=build --chown=app:app rise ./rise
COPY --from=build --chown=app:app rise_tasks ./rise_tasks
COPY --from=build --chown=app:app buildinfo.json buildinfo.json

EXPOSE 5000

ENTRYPOINT ["bash", "start.sh"]
