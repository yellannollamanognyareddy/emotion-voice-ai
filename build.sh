#!/usr/bin/env bash
pip install -r requirements.txt
python config/manage.py collectstatic --noinput
