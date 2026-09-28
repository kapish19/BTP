#!/usr/bin/env bash
# ==============================================================================
# Automated Dataset Downloader & Verifier for DG-ReID
# Supports: Market-1501, MSMT17, CUHK-SYSU, CUHK03, CUHK03-NP Protocol
# ==============================================================================

set -e

DATA_DIR="${1:-./data}"

echo "======================================================================"
echo "DG-ReID: Automated Dataset Setup from Kaggle"
echo "Target directory: ${DATA_DIR}"
echo "======================================================================"

# 1. Check if kaggle python package is available
if ! command -v kaggle &> /dev/null && ! python3 -m kaggle --version &> /dev/null; then
    echo "[!] kaggle CLI not found. Installing via pip..."
    pip install --upgrade kaggle
fi

# 2. Check for kaggle.json in standard locations
if [ -f "./kaggle.json" ]; then
    echo "[+] Found kaggle.json in current directory. Setting KAGGLE_CONFIG_DIR..."
    export KAGGLE_CONFIG_DIR="$(pwd)"
    chmod 600 ./kaggle.json
elif [ -f "${HOME}/.kaggle/kaggle.json" ]; then
    echo "[+] Found kaggle.json in ~/.kaggle/"
    chmod 600 "${HOME}/.kaggle/kaggle.json"
elif [ -n "${KAGGLE_USERNAME}" ] && [ -n "${KAGGLE_KEY}" ]; then
    echo "[+] Found KAGGLE_USERNAME and KAGGLE_KEY environment variables."
else
    echo "[-] ERROR: Kaggle credentials not found!"
    echo "    Please place your 'kaggle.json' API token in ~/.kaggle/kaggle.json"
    echo "    or in the current repository directory."
    echo "    To get a token: Kaggle -> Settings -> API -> Create New Token"
    exit 1
fi

mkdir -p "${DATA_DIR}"

# 3. Run automated python downloader for all benchmarks + protocols
python3 src/data/prepare_datasets.py --download all --dir "${DATA_DIR}"

# 4. Verify directory integrity
echo ""
echo "======================================================================"
echo "Verifying downloaded datasets..."
echo "======================================================================"
python3 src/data/prepare_datasets.py --verify all --dir "${DATA_DIR}"

echo ""
echo "[✓] All datasets downloaded, organized, and verified in ${DATA_DIR}."
