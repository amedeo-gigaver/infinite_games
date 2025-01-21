import copy
from abc import ABC, abstractmethod

import bittensor as bt

# Sync calls set weights and also resyncs the metagraph.
from neurons.miner.utils.config import add_args, check_config, config
from neurons.miner.utils.misc import ttl_get_block


def get_wallet(config: "bt.Config"):
    return bt.wallet(config=config)


def get_subtensor(config: "bt.Config"):
    return bt.subtensor(config=config)


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        # Set up logging with the provided configuration and directory.
        bt.logging(config=self.config, logging_dir=self.config.full_path)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = get_wallet(self.config)
        bt.logging.info(f"Wallet: {self.wallet} {self.wallet.hotkey.ss58_address}")

        # The subtensor is our connection to the Bittensor blockchain.
        self.subtensor = get_subtensor(self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
        )
        self.step = 0

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        ...

    @abstractmethod
    def run(self):
        ...

    def sync(self, set_weights_enabled=True):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        bt.logging.trace("Check neuron registration..")
        self.check_registered()

        bt.logging.trace("Check if need to sync metagraph...")
        if self.should_sync_metagraph():
            self.resync_metagraph()

        bt.logging.trace("Check if need to set weights..")
        if set_weights_enabled and self.should_set_weights():
            bt.logging.info("********* SUBMIT WEIGHTS *********")
            self.set_weights()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (self.block - self.metagraph.last_update[self.uid]) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            bt.logging.info(
                "self.config.neuron.disable_set_weights enabled from cli: skip set weight"
            )
            return False

        # Define appropriate logic for when set weights.
        should_set_weight = (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length
        bt.logging.trace(
            f"Weight / Current: {self.block}, Last : {self.metagraph.last_update[self.uid]} Epoch length: {self.config.neuron.epoch_length}, should set: {should_set_weight}"
        )
        return should_set_weight
