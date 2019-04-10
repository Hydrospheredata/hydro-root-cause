FROM python:3.7.2-alpine3.9

# TODO Fix pip modules versioning
RUN apk add --no-cache \
        --virtual=.build-dependencies \
        g++ gfortran file binutils \
        musl-dev python3-dev openblas-dev && \
    apk add libstdc++ openblas && \
    \
    ln -s locale.h /usr/include/xlocale.h && \
    \
    pip install numpy && \
    pip install pandas && \
    pip install scipy && \
    pip install scikit-learn && \
    \
    rm -r /root/.cache && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -r '{}' + && \
    find /usr/lib/python3.*/site-packages/ -name '*.so' -print -exec sh -c 'file "{}" | grep -q "not stripped" && strip -s "{}"' \; && \
    \
    rm /usr/include/xlocale.h && \
    \
    apk del .build-dependencies


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app

RUN pip install anchor2/ && pip install rise/

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]