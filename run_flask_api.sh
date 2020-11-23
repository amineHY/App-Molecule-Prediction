#!/bin/bash

export FLASK_APP=servier/src/flask_api.py
export FLASK_ENV=development

flask run --port 5000