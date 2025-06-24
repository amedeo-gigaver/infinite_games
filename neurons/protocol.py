from bittensor import Synapse


class EventPredictionSynapse(Synapse):
    events: dict = {
        # Example Shape
        # "market_type-event_id": {
        #     "event_id",
        #     "market_type",
        #     "probability",
        #     "reasoning",
        #     "miner_answered",
        #     "description",
        #     "cutoff",
        #     "metadata"
        # }
    }
