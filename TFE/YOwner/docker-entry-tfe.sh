#!/bin/bash

echo "--> starting tfe_benchmark..."
export PYTHONPATH=$PYTHONPATH:./:/app:/app/tfe_benchmark
printenv

echo "--> start tfe_server -->"
python3 -m tf_encrypted.player --config /app/config.json YOwner
echo "-------->"



