# -- DO NOT TOUCH BELOW - ENV SET --
# flake8: noqa: E402
import os
import sys

# Force torch - must be set before importing bittensor
os.environ["USE_TORCH"] = "1"

# Add the parent directory of the script to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
# -- DO NOT TOUCH ABOVE --

import time

from bittensor import logging

from neurons.miner.main import Miner

if __name__ == "__main__":
    start_time = time.time()
    with Miner() as miner:
        while True:
            logging.debug(f"Miner running for {time.time() - start_time:.1f} seconds")
            time.sleep(5)
