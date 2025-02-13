#!/bin/bash
set -euo pipefail

# Make sure these environment variables are set before running the script
: "${MNEMONIC_COLDKEY:?Must set MNEMONIC_COLDKEY}"
: "${MNEMONIC_HOTKEY:?Must set MNEMONIC_HOTKEY}"

# -- Regenerate coldkey --
btcli w regen_coldkey --wallet-path ~/.bittensor/wallets/ \
    --wallet-name validator --mnemonic "${MNEMONIC_COLDKEY}"

# -- Regenerate hotkey --
btcli w regen_hotkey --wallet-name validator --wallet-hotkey default \
    --mnemonic "${MNEMONIC_HOTKEY}"

echo "Coldkey and hotkey regenerated successfully."
