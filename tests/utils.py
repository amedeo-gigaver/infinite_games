from datetime import datetime, timedelta, timezone
from typing import List

from infinite_games.events.base import ProviderEvent
from infinite_games.protocol import EventPredictionSynapse


def after(**kwargs):
    return datetime.now(timezone.utc) + timedelta(**kwargs)


def before(**kwargs):
    return datetime.now(timezone.utc) - timedelta(**kwargs)


def fake_synapse_response(events: List[ProviderEvent]):
    uid_responses = []
    for uid in range(0, 256):
        synapse = EventPredictionSynapse()
        synapse.init(events)
        uid_responses.append(synapse)

    return uid_responses
