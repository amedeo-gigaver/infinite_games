import argparse

from bittensor.core.config import Config
from bittensor.core.subtensor import Subtensor
from bittensor.utils.btlogging import LoggingMachine
from bittensor_wallet.wallet import Wallet


def get_config() -> Config:
    # Build parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--netuid", type=int, help="Subnet netuid", choices=[6, 155], required=False
    )
    Subtensor.add_args(parser=parser)
    LoggingMachine.add_args(parser=parser)
    Wallet.add_args(parser=parser)

    # Validate args
    args = parser.parse_args()
    netuid = args.__getattribute__("netuid")
    network = args.__getattribute__("subtensor.network")

    if not (netuid == 6 and network == "finney") and not (netuid == 155 and network == "test"):
        raise ValueError(f"Invalid netuid {netuid} and network {network} combination.")

    return Config(parser=parser, strict=True)
