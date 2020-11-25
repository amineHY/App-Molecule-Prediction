#!/bin/bash

export FLASK_APP=app/src/flask_api.py
export FLASK_ENV=development

flask run --port 5000