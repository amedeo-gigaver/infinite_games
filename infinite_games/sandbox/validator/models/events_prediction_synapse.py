from bittensor import Synapse


class EventsPredictionSynapse(Synapse):
    events: dict

    def __innit__(self, events: dict):
        self.events = events
