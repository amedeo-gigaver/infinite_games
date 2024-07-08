# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

from datetime import datetime
import sys
from time import sleep
import torch
import unittest
import bittensor as bt

from infinite_games.events.base import EventStatus, ProviderEvent
from infinite_games.protocol import EventPredictionSynapse
from neurons.validator import Validator
from bittensor.mock import wallet_mock
from bittensor.mock.wallet_mock import MockWallet

from tests.utils import after, fake_synapse_response


class TestTemplateValidatorNeuronTestCase:

    def next_run(self, v: Validator):
        """Imitate Validator.run with 1 step"""
        # v.sync(False)

        bt.logging.info(
            f"Running validator {v.axon} on network: {v.config.subtensor.chain_endpoint} with netuid: {v.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {v.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        bt.logging.info(f"step({v.step}) block({v.block})")

        # Run multiple forwards concurrently.
        v.loop.run_until_complete(v.concurrent_forward())

        # Check if we should exit.

        # Sync metagraph and potentially set weights.
        v.sync()

        v.step += 1

    def test_validator_clean_run(self, mock_network, caplog):
        wallet, subtensor = mock_network
        v = Validator()

        # await v.forward()

        v.run_in_background_thread()

        sleep(6)

        # check from simple outputs for now.
        assert 'Provider initialized..' in caplog.text, 'Event Provider has to be initialized!'
        assert 'Event for submission: 0' in caplog.text, 'There should be no events initially!'
        assert 'EventAggregator Start watcher...' in caplog.text, 'Event watcher should be started'
        assert 'Processed miner responses.' not in caplog.text, 'There should not be any miner submissions!'
        v.stop_run_thread()
        assert v.event_provider
        assert v.event_provider.integrations
        assert len(v.event_provider.registered_events) > 0

    async def test_validator_settled_event_scores(self, mock_network, caplog, monkeypatch, disable_event_updates):
        wallet, subtensor = mock_network
        v = Validator()

        # await v.forward()

        self.next_run(v)
        # await restarted_vali.initialize_provider()
        # sleep(4)
        # v.stop_run_thread()
        test_event = ProviderEvent(
            '0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000',
            'azuro', 'Test event', after(hours=4), None,
            None, datetime.now(), EventStatus.PENDING,
            {},
            {
                'conditionId': 'conditionid',
                'slug': 'soccer-game-slug',
                'league': 'league'
            }
        )
        assert v.event_provider.register_event(test_event) is True
        assert v.event_provider.registered_events.get(f'{test_event.market_type}-{test_event.event_id}')
        assert len(v.event_provider.registered_events) == 1
        assert v.event_provider.integrations
        mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
        mock_response[3].events[f'{test_event.market_type}-{test_event.event_id}']['probability'] = 0.7
        mock_response[4].events[f'{test_event.market_type}-{test_event.event_id}']['probability'] = 0.9
        monkeypatch.setattr('neurons.validator.query_miners', lambda a, b, c: mock_response)

        self.next_run(v)
        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.update_event(test_event)
        assert v.scores[2] == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective brier score -> moving average
        assert round(v.scores[3].item(), 4) == 0.3640
        assert round(v.scores[4].item(), 4) == 0.3960

        assert round(v.average_scores[3].item(), 2) == 0.91
        assert round(v.average_scores[4].item(), 2) == 0.99


        test_event_2 = ProviderEvent(
            '0x9f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000',
            'azuro', 'Test event 2', after(hours=4), None,
            None, datetime.now(), EventStatus.PENDING,
            {},
            {
                'conditionId': 'conditionid2',
                'slug': 'soccer-game-slug2',
                'league': 'league2'
            }
        )
        assert v.event_provider.register_event(test_event_2) is True
        assert v.event_provider.registered_events.get(f'{test_event_2.market_type}-{test_event_2.event_id}')
        assert len(v.event_provider.registered_events) == 1
        assert v.event_provider.integrations
        mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
        mock_response[3].events[f'{test_event_2.market_type}-{test_event_2.event_id}']['probability'] = 0.6
        mock_response[4].events[f'{test_event_2.market_type}-{test_event_2.event_id}']['probability'] = 0.6
        monkeypatch.setattr('neurons.validator.query_miners', lambda a, b, c: mock_response)

        self.next_run(v)
        test_event_2.status = EventStatus.SETTLED
        test_event_2.answer = 1
        v.event_provider.update_event(test_event_2)
        assert v.scores[2] == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective brier score -> moving average
        assert round(v.scores[3].item(), 4) == 0.5544
        assert round(v.scores[4].item(), 4) == 0.5736

        assert round(v.average_scores[3].item(), 4) == 0.8750
        assert round(v.average_scores[4].item(), 4) == 0.9150
        # restarted_vali = Validator()

        # assert len(restarted_vali.event_provider.registered_events) > 0, 'Validator should have saved some new events from previous run'
