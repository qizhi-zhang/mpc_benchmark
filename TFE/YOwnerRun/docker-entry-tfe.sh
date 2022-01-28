#!/bin/bash

echo "--> starting tfe_benchmark..."
export PYTHONPATH=$PYTHONPATH:./:/app:/app/tfe_benchmark
printenv

echo "--> start tfe_server -->"
python3 -m tf_encrypted.player --config /app/config.json YOwner &
echo "--------> sleep 10--------"
sleep 10
echo "-------train lr on credit10 --------"
cd /app/tfe_benchmark
python3 train_and_predict_lr.py
echo "------------end of training of lr on credit10---------------"



