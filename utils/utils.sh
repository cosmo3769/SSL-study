#!/bin/bash

PYCMD=$(cat <<EOF
import wandb
wandb.login()
EOF
)

python -c "$PYCMD"