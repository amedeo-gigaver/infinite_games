# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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


import bittensor as bt
from typing import Optional, List

from infinite_games.events.base import ProviderEvent


class EventPredictionSynapse(bt.Synapse):
    # Dictionary of predictions. The keys are event provider ids
    # IDs, the values are the probability of the *second* result
    # occurring, so certainty that the first result occurs would be 0.0

    events: dict = {}

    def init(self, events: List[ProviderEvent]):
        self.events = {}
        for event in events:
            self.events[f'{event.market_type}-{event.event_id}'] = {
                "event_id": event.event_id,
                "market_type": event.market_type,
                'probability': None,
                "description": event.description,
                "starts": int(event.starts.timestamp()) if event.starts else None,
                "resolve_date": int(event.resolve_date.timestamp()) if event.resolve_date else None,
                "end_date": int(event.metadata['end_date']) if event.metadata.get('end_date') else None,
            }
