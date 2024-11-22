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
from typing import List

from infinite_games.events.base import ProviderEvent


class EventPredictionSynapse(bt.Synapse):

    events: dict = {}

    def init(self, events: List[ProviderEvent]):
        self.events = {}
        for event in events:
            market_type = (event.metadata.get('market_type', event.market_type) or '').lower()
            cutoff = event.metadata.get('cutoff')
            self.events[f'{event.market_type}-{event.event_id}'] = {
                "event_id": event.event_id,
                "market_type": market_type,
                'probability': None,
                "description": event.description,
                "cutoff": cutoff,
                "starts": int(cutoff) if (cutoff and market_type == 'azuro') else None,
                "resolve_date": int(event.resolve_date.timestamp()) if event.resolve_date else None,
                "end_date": int(event.metadata['end_date']) if event.metadata.get('end_date') else None,
            }
