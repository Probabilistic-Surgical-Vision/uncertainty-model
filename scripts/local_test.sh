#!/bin/bash

. ./venv/bin/activate

pip install -q --upgrade pip
pip install -q -r requirements.txt

python main.py config.yml da-vinci -e 2 -l adversarial \
    --training-size 16 --validation-size 16 \
    --save-model-to trained/da-vinci --save-model-every 1 \
    --save-evaluation-to results/da-vinci --evaluate-every 1 \
    --home ../ --no-cuda