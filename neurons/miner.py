# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
import os
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import typing
from datetime import datetime

import bittensor as bt

import infinite_games

# import base miner class which takes care of most of the boilerplate
from infinite_games.base.miner import BaseMinerNeuron
from infinite_games.events.azuro import AzuroProviderIntegration
from infinite_games.events.polymarket import PolymarketProviderIntegration
from infinite_games.utils.miner_cache import MinerCache, MinerCacheStatus, MinerCacheObject, MarketType

if os.getenv("OPENAI_KEY"):
    from llm.forecasting import Forecaster


class Miner(BaseMinerNeuron):
    """
    Miner neuron class. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    prev_emission = None

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.providers_set = False
        self.azuro = None
        self.polymarket = None
        self.cache = MinerCache()
        self.cache.initialize_cache()
        self.llm = Forecaster() if os.getenv("OPENAI_KEY") else None

    async def initialize_providers(self):
        self.azuro = await AzuroProviderIntegration()._ainit()
        self.polymarket = await PolymarketProviderIntegration()._ainit()

    async def _generate_prediction(self, market: MinerCacheObject) -> None:
        try:
            # LLM
            llm_prediction = (await self.llm.get_prediction(market)) if self.llm else None
            if llm_prediction is not None:
                market.event.probability = llm_prediction
                bt.logging.info(
                    "Assign llm prediction {} to event {} ".format(market.event.probability, market.event.event_id)
                )
                return

            # Polymarket
            if market.event.market_type == MarketType.POLYMARKET and self.polymarket is not None:
                x = await self.polymarket.get_event_by_id(market.event.event_id)
                market.event.probability = x["tokens"][0]["price"]
                bt.logging.info(
                    "Assign {} prob to polymarket event {}".format(market.event.probability, market.event.event_id)
                )
            # Azuro
            elif market.event.market_type == MarketType.AZURO and self.azuro is not None:
                x = await self.azuro.get_event_by_id(market.event.event_id)
                market.event.probability = 1.0 / float(x["outcome"]["currentOdds"])
                bt.logging.info(
                    "Assign {} prob to azuro event {}".format(market.event.probability, market.event.event_id)
                )
        except Exception as e:
            bt.logging.error("Failed to assign, probability, {}".format(e))

    async def forward(
            self, synapse: infinite_games.protocol.EventPredictionSynapse
    ) -> infinite_games.protocol.EventPredictionSynapse:
        """
        Processes the incoming synapse and attaches the response to the synapse.
        """
        if not self.providers_set:
            self.providers_set = True
            await self.initialize_providers()

        bt.logging.info("Incoming Events {}".format(len(synapse.events.items())))

        for cid, market in synapse.events.items():
            cached_market: typing.Optional[MinerCacheObject] = await self.cache.get(cid)
            if cached_market is not None:
                if cached_market.status == MinerCacheStatus.COMPLETED:
                    market["probability"] = cached_market.event.probability
                    bt.logging.info("Assign cache {} prob to polymarket event {}".format(cached_market.event.probability, cached_market.event.event_id))
                    # bt.logging.info("{} Assign cache {} prob to polymarket event {}".format(synapse.dendrite.hotkey, cached_market.event.probability, cached_market.event.event_id))
            else:
                new_market = MinerCacheObject.init_from_market(market)
                await self.cache.add(cid, self._generate_prediction, new_market)

        return synapse

    async def blacklist(
        self, synapse: infinite_games.protocol.EventPredictionSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.EventPredictionSynapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        # TODO(developer): Define how miners should blacklist requests.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.debug(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.debug(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: infinite_games.protocol.EventPredictionSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.EventPredictionSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.debug(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    def save_state(self):
        pass


bt.debug(True)
# bt.trace(true)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
