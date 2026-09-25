#!/bin/bash
echo "Setting up ULTIMATE MQ Testbench environment..."

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Downgrade pip and setuptools to allow hamamatsu's bad formed entry point
echo "Downgrading pip to bypass hamamatsu strict packaging rules..."
pip install "pip<24.0" "setuptools<69.0"

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt

# Patch pylablib for Linux Hamamatsu compatibility
echo "Patching pylablib for Linux DCAM-API..."
sed -i 's/"dcamapi.dll"/"libdcamapi.so"/g' .venv/lib/python*/site-packages/pylablib/devices/DCAM/dcamapi4_lib.py

echo "Done! Run 'source .venv/bin/activate' to start."