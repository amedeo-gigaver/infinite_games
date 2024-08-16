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

import asyncio
from datetime import datetime, timedelta, timezone
import sys
from time import sleep
from freezegun import freeze_time
import torch
import unittest
import bittensor as bt

from infinite_games.events.azuro import AzuroProviderIntegration
from infinite_games.events.base import CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES, EventStatus, ProviderEvent
from infinite_games.events.polymarket import PolymarketProviderIntegration
from infinite_games.protocol import EventPredictionSynapse
from neurons.validator import Validator
from bittensor.mock import wallet_mock
from bittensor.mock.wallet_mock import MockWallet

from tests.providers import MockAcledProviderIntegration, MockAzuroProviderIntegration, MockPolymarketProviderIntegration
from tests.utils import after, before, fake_synapse_response


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

    # def test_validator_clean_run(self, mock_network, caplog):
    #     wallet, subtensor = mock_network
    #     v = Validator(integrations=[
    #         AzuroProviderIntegration(),
    #         PolymarketProviderIntegration()
    #     ])

    #     # await v.forward()

    #     v.run_in_background_thread()

    #     sleep(12)

    #     # check from simple outputs for now.
    #     assert 'Provider initialized..' in caplog.text, 'Event Provider has to be initialized!'
    #     assert 'Event for submission: 0' in caplog.text, 'There should be some events!'
    #     # assert 'EventAggregator Start watcher...' in caplog.text, 'Event watcher should be started'
    #     assert 'Processed miner responses.' not in caplog.text, 'There should not be any miner submissions!'
    #     v.stop_run_thread()
    #     assert v.event_provider
    #     assert v.event_provider.integrations
    #     assert len(v.event_provider.registered_events) > 0, "There should be at least one registered event"

    async def test_validator_acled_events(
            self, mock_network, caplog, monkeypatch
    ):
        wallet, subtensor = mock_network
        acled_provider = MockAcledProviderIntegration()
        v = Validator(integrations=[
            acled_provider
        ])

        # await v.forward()
        print('First run')
        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date):
            self.next_run(v)
            # await restarted_vali.initialize_provider()
            # sleep(4)
            # v.stop_run_thread()
            sleep(2)
            print('Second run')
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events['acled-dbcba93a-fe3b-4092-b918-8231b23f2faa']['probability'] = 1
            mock_response[4].events['acled-dbcba93a-fe3b-4092-b918-8231b23f2faa']['probability'] = 1
            monkeypatch.setattr('neurons.validator.query_miners', lambda a, b, c: mock_response)
            self.next_run(v)
        for window in range(1, 42):

            window_time = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * window)
            with freeze_time(window_time):
                self.next_run(v)

        # based on providers.py hardcode values
        settle_date = initial_date + timedelta(days=7)
        v.event_provider.registered_events['acled-7b787c68-d6df-4138-a10b-0de76eeec5c3']
        with freeze_time(settle_date):

            test_event = await acled_provider.get_single_event('dbcba93a-fe3b-4092-b918-8231b23f2faa')

            test_event.status = EventStatus.SETTLED
            test_event.answer = 1
            v.event_provider.update_event(test_event)

            self.next_run(v)

        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.4, 0.4)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (0.5, 0.5)

    async def test_validator_settled_event_scores_polymarket_aggregation_interval(
            self, mock_network, caplog, monkeypatch, disable_event_updates
    ):
        wallet, subtensor = mock_network
        v = Validator(integrations=[
            MockAzuroProviderIntegration(max_pending_events=6),
            MockPolymarketProviderIntegration()
        ])

        # await v.forward()
        print('First run')
        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date):
            self.next_run(v)
            # await restarted_vali.initialize_provider()
            # sleep(4)
            # v.stop_run_thread()
            test_event = ProviderEvent(
                '0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000',
                datetime.now(timezone.utc),
                'polymarket', 'Test event 1', None, after(hours=12),
                None, datetime.now(timezone.utc), EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
                }
            )
            assert v.event_provider.register_event(test_event) is True
            assert v.event_provider.registered_events.get(f'{test_event.market_type}-{test_event.event_id}')
            assert len(v.event_provider.registered_events) == 1
            assert v.event_provider.integrations
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events[f'{test_event.market_type}-{test_event.event_id}']['probability'] = 0.7
            mock_response[4].events[f'{test_event.market_type}-{test_event.event_id}']['probability'] = 0.5
            monkeypatch.setattr('neurons.validator.query_miners', lambda a, b, c: mock_response)
            print('Second run')
            self.next_run(v)
            mock_response = fake_synapse_response(v.event_provider.get_events_for_submission())
            mock_response[3].events[f'{test_event.market_type}-{test_event.event_id}']['probability'] = 0.5
            mock_response[4].events[f'{test_event.market_type}-{test_event.event_id}']['probability'] = 0.9
            monkeypatch.setattr('neurons.validator.query_miners', lambda a, b, c: mock_response)
            self.next_run(v)

        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window):
            self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window):
            self.next_run(v)

        fourth_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 3)
        with freeze_time(fourth_window):
            self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.update_event(test_event)
        assert v.scores[2] == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective brier score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.0221, 0.7779)
        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (0.028, 0.972)

    async def test_validator_settled_event_scores_polymarket_short(self, mock_network, caplog, monkeypatch, disable_event_updates):
        wallet, subtensor = mock_network
        v = Validator(integrations=[
            MockAzuroProviderIntegration(max_pending_events=6), 
            MockPolymarketProviderIntegration()
        ])

        # await v.forward()
        print('First run')
        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date):
            self.next_run(v)
            # await restarted_vali.initialize_provider()
            # sleep(4)
            # v.stop_run_thread()
            test_event = ProviderEvent(
                '0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000',
                datetime.now(timezone.utc),
                'polymarket', 'Test event 1', None, after(hours=12),
                None, datetime.now(timezone.utc), EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
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
            print('Second run')
            self.next_run(v)

        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window):
            self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window):
            self.next_run(v)

        fourth_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 3)
        with freeze_time(fourth_window):
            self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.update_event(test_event)
        assert v.scores[2] == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective brier score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.0954, 0.7046)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (0.119, 0.881)

    async def test_validator_settled_event_scores_polymarket_longer_settle_date(self, mock_network, caplog, monkeypatch, disable_event_updates):
        wallet, subtensor = mock_network
        v = Validator(integrations=[
            MockAzuroProviderIntegration(max_pending_events=6),
            MockPolymarketProviderIntegration()
        ])

        # await v.forward()
        print('First run')
        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date):
            self.next_run(v)
            # await restarted_vali.initialize_provider()
            # sleep(4)
            # v.stop_run_thread()
            test_event = ProviderEvent(
                '0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000',
                datetime.now(timezone.utc),
                'polymarket', 'Test event 1', None, after(days=12),
                None, datetime.now(timezone.utc), EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
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
            print('Second run')
            self.next_run(v)

        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window):
            self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window):
            self.next_run(v)

        fourth_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 10)
        with freeze_time(fourth_window):
            self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.update_event(test_event)
        assert v.scores[2] == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective brier score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.0, 0.0)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3)) == (0.0, 0.0)

    async def test_validator_settled_event_scores_polymarket_earlier_settle_date(self, mock_network, caplog, monkeypatch, disable_event_updates):
        wallet, subtensor = mock_network
        v = Validator(integrations=[
            MockAzuroProviderIntegration(max_pending_events=6), 
            MockPolymarketProviderIntegration()
        ])

        # await v.forward()
        print('First run')
        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date):
            self.next_run(v)
            # await restarted_vali.initialize_provider()
            # sleep(4)
            # v.stop_run_thread()
            test_event = ProviderEvent(
                '0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000',
                datetime.now(timezone.utc),
                'polymarket', 'Test event 1', None, after(days=12),
                None, datetime.now(timezone.utc), EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
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
            print('Second run')
            self.next_run(v)

        second_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
        with freeze_time(second_window):
            self.next_run(v)
        third_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 2)
        with freeze_time(third_window):
            self.next_run(v)

        tenth_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 10)
        with freeze_time(tenth_window):
            self.next_run(v)

        eleventh_window = initial_date + timedelta(minutes=CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES * 11)
        with freeze_time(eleventh_window):
            # self.next_run(v)

            test_event.status = EventStatus.SETTLED
            test_event.answer = 1
            v.event_provider.update_event(test_event)
        assert v.scores[2] == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective brier score -> moving average
        assert round(v.scores[3].item(), 4) == 0.0
        assert round(v.scores[4].item(), 4) == 0.0

        assert round(v.average_scores[3].item(), 3) == 0.0
        assert round(v.average_scores[4].item(), 3) == 0.0

    async def test_validator_settled_event_scores_azuro(self, mock_network, caplog, monkeypatch, disable_event_updates):
        wallet, subtensor = mock_network
        v = Validator(integrations=[
            MockAzuroProviderIntegration(max_pending_events=6),
            MockPolymarketProviderIntegration()
        ])

        # await v.forward()
        print('First run')
        initial_date = datetime(year=2024, month=1, day=3)
        with freeze_time(initial_date):
            self.next_run(v)
            # await restarted_vali.initialize_provider()
            # sleep(4)
            # v.stop_run_thread()
            test_event = ProviderEvent(
                '0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000',
                datetime.now(timezone.utc),
                'azuro', 'Test event 1', after(hours=12), None,
                None, datetime.now(timezone.utc), EventStatus.PENDING,
                {},
                {
                    # 'conditionId': 'conditionid',
                    # 'slug': 'soccer-game-slug',
                    # 'league': 'league'
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
            print('Second run')
            self.next_run(v)

        test_event.status = EventStatus.SETTLED
        test_event.answer = 1
        v.event_provider.update_event(test_event)
        assert v.scores[2] == 0.0
        # 0.7, 0.9
        # uid 3 and 4 calculated based on respective brier score -> moving average
        assert (round(v.scores[3].item(), 4), round(v.scores[4].item(), 4)) == (0.0954, 0.7046)

        assert (round(v.average_scores[3].item(), 3), round(v.average_scores[4].item(), 3))  == (0.119, 0.881)
