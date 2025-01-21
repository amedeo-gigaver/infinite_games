from unittest.mock import Mock

import torch

from neurons.miner.main import Miner


class MockMinerWithActualBlacklistMethod(Miner):
    def __init__(self):
        self.metagraph = Mock()
        self.subtensor = Mock()
        self.subtensor.network = "mock"
        self.metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
        self.metagraph.validator_permit = [True, True, False]
        self.metagraph.S = torch.tensor([10000.0, 2000.0, 5.1])

    async def blacklist(self, synapse):
        return await super().blacklist(synapse)
