from bittensor.core.dendrite import DendriteMixin
from bittensor.core.metagraph import MetagraphMixin

from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.models.events_prediction_synapse import (
    EventsPredictionSynapse,
)
from infinite_games.sandbox.validator.scheduler.task import AbstractTask


class QueryMiners(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    dendrite: DendriteMixin
    metagraph: MetagraphMixin

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        dendrite: DendriteMixin,
        metagraph: MetagraphMixin,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate dendrite
        if not isinstance(dendrite, DendriteMixin):
            raise TypeError("dendrite must be an instance of DendriteMixin.")

        # Validate metagraph
        if not isinstance(metagraph, MetagraphMixin):
            raise TypeError("metagraph must be an instance of MetagraphMixin.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.dendrite = dendrite
        self.metagraph = metagraph

    @property
    def name(self):
        return "query-miners"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Get events to predict
        events = await self.db_operations.get_events_to_predict()

        if not len(events):
            return

        # Build synapse
        synapse = self.make_predictions_synapse(events)

        # Get axons to query
        axons = self.get_axons()

        # Query miners
        predictions = self.query_miners(axons=axons, synapse=synapse)

        # TODO, parse predictions
        # Store predictions
        await self.db_operations.upsert_predictions(predictions=predictions)

    def get_axons(self):
        pass

    def query_miners(self, axons: any, synapse: EventsPredictionSynapse):
        timeout = 120

        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=axons,
            synapse=synapse,
            # Do not deserialize the response so that we have access to the raw response.
            deserialize=False,
            timeout=timeout,
        )

        return responses

    def make_predictions_synapse(self, events: tuple[any]) -> EventsPredictionSynapse:
        events = {}

        for event in events:
            event_id = event[0]
            market_type = event[1]
            description = event[2]
            cutoff = event[3]
            resolve_date = event[4]
            end_date = event[5]
            metadata = event[6]

            market_type = (metadata.get("market_type", market_type)).lower()

            self.events[f"{market_type}-{event_id}"] = {
                "event_id": event_id,
                "market_type": market_type,
                "probability": None,
                "miner_answered": False,
                "description": description,
                "cutoff": cutoff,
                "starts": int(cutoff) if (cutoff and market_type == "azuro") else None,
                "resolve_date": (int(resolve_date.timestamp()) if resolve_date else None),
                "end_date": (end_date.timestamp() if end_date else None),
            }

        return EventsPredictionSynapse(events=events)
