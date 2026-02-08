#!/bin/bash

# 1. Activate Virtual Environment
source /home/arika/Documents/hacklahoma_26_facnet_demo/.venv/bin/activate

# 2. Set the Library Paths (The Magic Fix)
export LD_LIBRARY_PATH=/home/arika/Documents/hacklahoma_26_facnet_demo/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/arika/Documents/hacklahoma_26_facnet_demo/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/arika/Documents/hacklahoma_26_facnet_demo/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:/home/arika/Documents/hacklahoma_26_facnet_demo/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# 3. Set Display
export DISPLAY=:1

# 4. Run the App
python /home/arika/Documents/hacklahoma_26_facnet_demo/main.py
