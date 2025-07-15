import asyncio
import time
import typing
from datetime import datetime, timezone

import bittensor as bt

from neurons.miner.base.miner import BaseMinerNeuron
from neurons.miner.models.event import MinerEvent, MinerEventStatus
from neurons.miner.utils.storage import MinerStorage
from neurons.miner.utils.task_executor import TaskExecutor
from neurons.protocol import EventPredictionSynapse
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

VAL_MIN_STAKE = 10000
DEV_MINER_UID = 93


class Miner(BaseMinerNeuron):
    """
    Miner neuron class. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    prev_emission = None

    def __init__(self, logger: InfiniteGamesLogger, config=None, assign_forecaster=None):
        super(Miner, self).__init__(config=config)
        self.is_testnet = self.metagraph.network == "test"
        self.task_executor = TaskExecutor(logger=logger)
        self.storage = MinerStorage(logger=logger)
        self.assign_forecaster = assign_forecaster
        self.logger = logger

    async def initialize(self):
        try:
            await self.storage.load(
                condition=lambda event: event.cutoff > datetime.now(timezone.utc)
            )
            storage_task = asyncio.create_task(self.storage.save())
            executor_task = asyncio.create_task(self.task_executor.execute())
            self.storage_task, self.task_executor_task = storage_task, executor_task
            self.logger.info(
                "Miner {} initialized on network: {}: testnet {}".format(
                    self.uid, self.metagraph.network, self.is_testnet
                )
            )
        except Exception:
            bt.logging.error("Failed to initialize miner", exc_info=True)
            raise

    async def forward(self, synapse: EventPredictionSynapse) -> EventPredictionSynapse:
        """
        Processes the incoming synapse and attaches the response to the synapse.
        """
        start_time = time.time()

        self.logger.info(
            "Incoming Events {}, from {}".format(
                len(synapse.events.items()), synapse.dendrite.hotkey
            )
        )
        count = 0
        for event_key, validator_event in synapse.events.items():
            try:
                event: MinerEvent | None = await self.storage.get(event_key)
                if event is None:
                    self.logger.info(f"Event {event_key} is a new event")
                    event = MinerEvent.model_validate(validator_event.model_dump())

                    if event.cutoff > datetime.now(timezone.utc):
                        await self.storage.set(event_key, event)
                    else:
                        continue
                else:
                    self.logger.debug(f"Event {event_key} is already in storage")
            except Exception:
                self.logger.error(f"Failed to get/create event {event_key}", exc_info=True)
            else:
                status = event.get_status()
                if status == MinerEventStatus.UNRESOLVED:
                    forecaster = await self.assign_forecaster(event)
                    await self.task_executor.add_task(forecaster)
                    event.set_status(MinerEventStatus.PENDING)
                    self.logger.info(f"Event {event_key} is pending resolution")
                elif status == MinerEventStatus.RESOLVED:
                    probability = event.get_probability()
                    validator_event.probability = probability
                    validator_event.miner_answered = probability is not None

                    reasoning = event.get_reasoning()
                    validator_event.reasoning = reasoning

                    count += probability is not None
                    self.logger.info(
                        f"Event {event_key} is resolved with probability {probability}"
                    )

        self.logger.info(
            f"Miner answered on validator {synapse.dendrite.hotkey} in {time.time() - start_time:.2f} seconds "
            f"for {count}/{len(synapse.events.items())} events"
        )
        return synapse

    async def blacklist(self, synapse: EventPredictionSynapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead constructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (EventPredictionSynapse): A synapse object constructed from the headers of the incoming request.

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
        try:
            if self.subtensor.network in ["test", "mock", "local"]:
                return False, "Blacklisting disabled in testnet"

            # Check if the hotkey is provided
            if not synapse.dendrite.hotkey:
                return True, "Hotkey not provided"

            # Check if the hotkey is recognized
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                bt.logging.warning(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
                return True, "Unrecognized hotkey"

            # Check if the hotkey is a validator
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

            stake = self.metagraph.S[uid].item()
            bt.logging.debug(f"Validator {synapse.dendrite.hotkey} has stake {stake}")
            if stake < VAL_MIN_STAKE:
                bt.logging.warning(
                    f"Blacklisting a request from hotkey {synapse.dendrite.hotkey} with stake {stake}"
                )
                return True, "Low stake"

            bt.logging.debug(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
            return False, "Hotkey recognized!"
        except Exception as e:
            bt.logging.error(f"Failed to validate hotkey {synapse.dendrite.hotkey}: {e}")
            return True, "Failed to validate hotkey"

    async def priority(self, synapse: EventPredictionSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (EventPredictionSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # Get the caller index.
        priority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.

        bt.logging.debug(f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority)

        return priority
