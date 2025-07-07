#!/bin/bash

# Activate conda environment (optional, if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mm

# Run the FastAPI server
PYTHONUNBUFFERED=1 python ~/ace/mm/MM/agent/servers/serve.py \
    --host 0.0.0.0 --port 5000 > ~/ace/mm/experiments/server.log 2>&1