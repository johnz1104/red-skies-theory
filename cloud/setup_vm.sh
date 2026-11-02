#!/usr/bin/env bash
# Set up a fresh Ubuntu 24.04 VM (Python 3.12) to train without Docker.
# Run on the VM from the copied repository: bash cloud/setup_vm.sh
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y python3.12-venv tmux rsync
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
# the CPU build: on Linux (arm64 and x86_64 alike) PyPI's torch pulls the CUDA libraries,
# whose cuDNN and cuSPARSELt wheels alone are 0.7-0.9 GB
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' cloud/requirements.txt)"
.venv/bin/pip install -r cloud/requirements.txt
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m cold_war_sim.campaign catalog --check
echo "ready: $(nproc) cores"
