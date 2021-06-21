#!/usr/bin/env sh

echo "$1"

if [ -z "$1" ]; then
    echo "Choose mode: 'service' or 'worker'"
    exit 1
fi

if [ "$1" = 'service' ]; then
    python rootcause/app.py
elif [ "$1" = 'worker' ]; then
    celery -A rootcause.app.celery worker -l info -O fair -Q rootcause
else
    echo "'$1' mode is incorrect. Supported modes are 'service' or 'worker'"
    exit 1
fi