#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate the virtual environment and install requirements
source venv/bin/activate
pip install -r requirements.txt
pip install ./linux_packages/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
pip install ./linux_packages/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl

echo "Installation Complete"