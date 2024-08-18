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

import asyncio
from datetime import datetime, timedelta, timezone
import itertools
import logging
import math
import os

from infinite_games.events.acled import AcledProviderIntegration
os.environ['USE_TORCH'] = '1'
import time
import traceback

from infinite_games.utils.query import query_miners
import bittensor as bt
import torch
import infinite_games

# import base validator class which takes care of most of the boilerplate
from infinite_games.base.validator import BaseValidatorNeuron
from infinite_games.events.base import CLUSTER_EPOCH_2024, CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES, EventAggregator, EventStatus, ProviderEvent, ProviderIntegration, Submission
from infinite_games.events.azuro import AzuroProviderIntegration
from infinite_games.events.polymarket import PolymarketProviderIntegration


class Validator(BaseValidatorNeuron):
    """
    Validator neuron class.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, integrations, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.active_markets = {}
        self.blocktime = 0
        self.event_provider = None
        self.SEND_LOGS_INTERVAL = 60 * 60
        self.SEND_MINER_LOGS_INTERVAL = 60 * 60 * 4
        self.integrations = integrations

    async def initialize_provider(self):
        if not self.event_provider:
            self.event_provider: EventAggregator = await EventAggregator.create(
                state_path=self.config.neuron.full_path + '/events-v2.pickle',
                integrations=self.integrations
            )
            self.event_provider.load_state()
            self.event_provider.on_event_updated_hook(self.on_event_update)
            if os.getenv('VALIDATOR_WATCH_EVENTS_DISABLED', "0") == "0":
                # watch for existing registered events
                self.loop.create_task(self.event_provider.watch_events())
                # pull new markets
                self.loop.create_task(self.event_provider.collect_events())
            bt.logging.info(f'TARGET_MONITOR_HOTKEY: {os.environ.get("TARGET_MONITOR_HOTKEY", "None")}')
            bt.logging.info(f'GRAFANA_API_KEY: {os.environ.get("GRAFANA_API_KEY", "None")}')
            if self.wallet.hotkey.ss58_address == os.environ.get('TARGET_MONITOR_HOTKEY'):
                self.loop.create_task(self.send_stats())
                # self.loop.create_task(self.track_interval_stats())
            bt.logging.debug("Provider initialized..")

    async def send_stats(self):
        bt.logging.info('Scheduling sending average stats.')
        while True:

            all_uids = [uid for uid in range(self.metagraph.n.item())]
            bt.logging.debug(f"Sending daily average total: {self.average_scores}")
            self.send_average_scores(miner_scores=list(zip(all_uids, self.average_scores.tolist(), self.scores.tolist())))
            await asyncio.sleep(self.SEND_LOGS_INTERVAL)

    async def send_interval_stats(self):
        now = datetime.now(timezone.utc)
        minutes_since_epoch = int((now - CLUSTER_EPOCH_2024).total_seconds()) // 60
        # previous interval from current one filled already, sending it.
        interval_prev_start_minutes = minutes_since_epoch - (minutes_since_epoch % (CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)) - CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
        all_uids = [uid for uid in range(self.metagraph.n.item())]
        bt.logging.debug(f"Sending interval data: {interval_prev_start_minutes}")
        for uid in all_uids:
            metrics = []
            for event in self.event_provider.get_events_for_submission():
                predictions = event.miner_predictions
                prediction_intervals = predictions.get(uid)
                # bt.logging.info(prediction_intervals)
                if event.market_type == 'azuro':
                    if not prediction_intervals:
                        ans = -1
                    else:
                        ans = prediction_intervals[0]['total_score']
                        if ans is None:
                            ans = -1
                        else:
                            ans = max(0, min(1, ans))  # Clamp the answer
                else:

                    # self.event_provider._resolve_previous_intervals(pe, uid.item(), None)
                    if not prediction_intervals:
                        ans = -1
                    else:

                        interval_data = prediction_intervals.get(interval_prev_start_minutes, {
                            'total_score': None
                        })
                        ans: float = interval_data['total_score']
                        if ans is None:
                            ans = -1
                metrics.append([uid, event.event_id, event.market_type, interval_prev_start_minutes, ans ])
            self.send_interval_data(miner_data=metrics)
            await asyncio.sleep(2)

    async def track_interval_stats(self):
        bt.logging.info('Scheduling sending interval stats.')

        while True:
            await self.send_interval_stats()
            bt.logging.info(f'Waiting for next {self.SEND_MINER_LOGS_INTERVAL} seconds to schedule interval logs..')
            await asyncio.sleep(self.SEND_MINER_LOGS_INTERVAL)

    def on_event_update(self, pe: ProviderEvent):
        """Hook called whenever we have settling events. Event removed when we return True"""
        if pe.status == EventStatus.SETTLED:
            bt.logging.info(f'Settled event: {pe} {pe.description[:100]} answer: {pe.answer}')
            miner_uids = infinite_games.utils.uids.get_all_uids(self)
            correct_ans = pe.answer
            if correct_ans is None:
                bt.logging.info(f"Unknown answer for event, discarding : {pe}")
                return True

            predictions = pe.miner_predictions
            if not predictions:
                bt.logging.warning(f"No predictions for {pe} skipping..")
                return True
            integration: ProviderIntegration = self.event_provider.get_integration(pe)
            # if not integration:
            #     bt.logging.error(f'no integration found for event {pe}. will skip this event!')
            #     return True
            cutoff = integration.latest_submit_date(pe)
            bt.logging.info(f'Miners to update: {len(miner_uids)} submissions: {len(predictions.keys())} from {self.metagraph.n.item()}')
            bt.logging.info(f'Register: {pe.registered_date} cutoff: {cutoff} tz {cutoff.tzinfo}, resolve: {pe.resolve_date}')

            # we take either now or cutoff whatever is lowest(event can be settled earlier
            cutoff_minutes_since_epoch = int((cutoff - CLUSTER_EPOCH_2024).total_seconds()) // 60
            cutoff_interval_start_minutes = cutoff_minutes_since_epoch - (cutoff_minutes_since_epoch % CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
            now = datetime.now(timezone.utc)
            now_minutes_since_epoch = int((now - CLUSTER_EPOCH_2024).total_seconds()) // 60
            now_interval_start_minutes = now_minutes_since_epoch - (now_minutes_since_epoch % CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
            bt.logging.info(f'Comparing cutoff to now: {cutoff=}: {cutoff_interval_start_minutes}, {now=} {now_interval_start_minutes}')
            effective_finish_start_minutes = min(cutoff_interval_start_minutes, now_interval_start_minutes)
            start_minutes_since_epoch = int((pe.registered_date - CLUSTER_EPOCH_2024).total_seconds()) // 60
            start_interval_start_minutes = start_minutes_since_epoch - (start_minutes_since_epoch % CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
            total_intervals = (effective_finish_start_minutes - start_interval_start_minutes) // CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
            first_n_intervals = 1
            bt.logging.info(f'{integration.__class__.__name__} {first_n_intervals=} intervals: {pe.registered_date=} {effective_finish_start_minutes=} {pe.resolve_date=} {cutoff=}  total={total_intervals}')
            scores = []
            non_penalty_brier = []
            for uid in miner_uids:
                prediction_intervals = predictions.get(uid.item())
                # bt.logging.info(prediction_intervals)
                if pe.market_type == 'azuro':
                    if not prediction_intervals:
                        scores.append(0)
                        continue

                    ans = prediction_intervals[0]['total_score']
                    if ans is None:
                        scores.append(0)
                        continue
                    ans = max(0, min(1, ans))  # Clamp the answer
                    brier_score = 1 - ((ans - correct_ans)**2)
                    non_penalty_brier.append(brier_score)
                    scores.append(max(brier_score - 0.75, 0))
                    bt.logging.info(f'settled answer for {uid=} for {pe.event_id=} {ans=} {brier_score=}')
                else:

                    # self.event_provider._resolve_previous_intervals(pe, uid.item(), None)
                    if not prediction_intervals:
                        scores.append(0)
                        continue
                    mk = []

                    weights_sum = 0

                    for interval_start_minutes in range(start_interval_start_minutes, effective_finish_start_minutes, CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES):

                        interval_data = prediction_intervals.get(interval_start_minutes, {
                            'total_score': None
                        })
                        ans: float = interval_data['total_score']
                        current_interval_no = (interval_start_minutes - start_interval_start_minutes) // CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES
                        interval_start_date = CLUSTER_EPOCH_2024 + timedelta(minutes=interval_start_minutes)
                        if current_interval_no + 1 <= first_n_intervals:
                            wk = 1
                        else:

                            wk = math.exp(-(total_intervals/(total_intervals - current_interval_no )) + 1)
                        weights_sum += wk
                        # bt.logging.info(f'answer for {uid=} {interval_start_minutes=} {ans=} total={total_intervals} curr={current_interval_no} {wk=} ')
                        if ans is None:
                            mk.append(0)
                            continue
                        ans = max(0, min(1, ans))  # Clamp the answer
                        # bt.logging.debug(f'Submission of {uid=} {ans=}')
                        brier_score = 1 - ((ans - correct_ans)**2)
                        mk.append(wk * brier_score)

                        bt.logging.info(f'answer for {uid=} {interval_start_minutes=} {interval_start_date=} {ans=} total={total_intervals} curr={current_interval_no} {wk=} {brier_score=}')
                    final_avg_brier = sum(mk) / weights_sum
                    bt.logging.info(f'final avg brier answer for {uid=} {final_avg_brier=}')
                    # 1/2 does not bring any value, add penalty for that
                    non_penalty_brier.append(final_avg_brier)
                    penalty_brier_score = max(final_avg_brier - 0.75 , 0)

                    scores.append(penalty_brier_score)
            brier_scores = torch.FloatTensor(non_penalty_brier)
            scores = torch.FloatTensor(scores)
            if all(score.item() <= 0.0 for score in scores):
                # bt.logging.info('All effective scores zero for this event!')
                pass
            else:
                alpha = 0
                beta = 1
                non_zeros = scores != 0
                scores[non_zeros] = alpha * scores[non_zeros] + (beta * torch.exp(25*scores[non_zeros]))
            bt.logging.info(f'With exp scores {scores}')
            scores = torch.nn.functional.normalize(scores, p=1, dim=0)
            bt.logging.info(f'Normalized {scores}')
            self.update_scores(scores, miner_uids)
            self.send_event_scores(zip(miner_uids, itertools.repeat(pe.market_type), itertools.repeat(pe.event_id), brier_scores, scores))
            return True
        elif pe.status == EventStatus.DISCARDED:
            bt.logging.info(f'Canceled event: {pe} removing from registry!')
            self.event_provider.remove_event(pe)

        return False

    async def forward(self):
        """
        The forward function is called by the validator every time step.

        """
        await self.initialize_provider()
        self.reset_daily_average_scores()
        self.print_info()
        block_start = self.block
        miner_uids = infinite_games.utils.uids.get_all_uids(self)
        # Create synapse object to send to the miner.
        synapse = infinite_games.protocol.EventPredictionSynapse()
        events_available_for_submission = self.event_provider.get_events_for_submission()
        bt.logging.info(f'Event for submission: {len(events_available_for_submission)}')
        synapse.init(events_available_for_submission)
        # print("Synapse body hash", synapse.computed_body_hash)
        bt.logging.info(f'Axons: {len(self.metagraph.axons)}')
        # for axon in self.metagraph.axons:
        #     bt.logging.info(f'IP: {axon.ip}, hotkey id: {axon.hotkey}')

        bt.logging.info("Querying miners..")
        # The dendrite client queries the network.
        responses = query_miners(self.dendrite, [self.metagraph.axons[uid] for uid in miner_uids], synapse)

        # synapse.events['azuro-0x7f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927405030000000000000357953524_142'] = {
        #     'event_id': '0x7f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927405030000000000000357953524_142',
        #     'probability': 0.7,
        #     'market_type': 'azuro'
        # }

        # Update answers
        miners_activity = set()
        now = datetime.now(timezone.utc)
        minutes_since_epoch = int((now - CLUSTER_EPOCH_2024).total_seconds()) // 60
        interval_start_minutes = minutes_since_epoch - (minutes_since_epoch % (CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES))

        for (uid, resp) in zip(miner_uids, responses):
            any_miner_processed = False
            # print(uid, resp)
            for (event_id, event_data) in resp.events.items():
                market_event_id = event_data.get('event_id')
                provider_name = event_data.get('market_type')
                score = event_data.get('probability')

                provider_event = self.event_provider.get_registered_event(provider_name, market_event_id)
                if not provider_event:
                    # bt.logging.warning(f'Miner submission for non registered event detected  {uid=} {provider_name=} {market_event_id=}')
                    continue
                # if uid != 4:
                #     continue
                if score is None:
                    # bt.logging.debug(f'uid: {uid.item()} no prediction for {event_id} sent, skip..')
                    continue
                integration = self.event_provider.integrations.get(provider_event.market_type)
                if not integration:
                    bt.logging.error(f'No integration found to register miner submission {uid=} {event_id=} {score=}')
                    continue
                if integration.available_for_submission(provider_event):
                    bt.logging.info(f'Submission {uid=} for {interval_start_minutes} {event_id}')
                    any_miner_processed = True
                    await self.event_provider.miner_predict(provider_event, uid.item(), score, interval_start_minutes, self.block)
                else:
                    # bt.logging.warning(f'Submission received, but this event is not open for submissions miner {uid=} {event_id=} {score=}')
                    continue
            # if len(miner_submitted) > 0:
                # bt.logging.info(f'Submission {uid=} for {interval_start_minutes} saved, events: {len(miner_submitted)}')

            # bt.logging.info(f'uid: {uid.item()} got prediction for events: {len(miner_submitted)}')

        if any_miner_processed:
            bt.logging.info("Processed miner responses.")
        else:
            bt.logging.info('No miner submissions received')
        self.blocktime += 1
        if os.environ.get('ENV') != 'pytest':
            while block_start == self.block:
                bt.logging.debug(f"FORWARD INTERVAL: {float(os.environ.get('VALIDATOR_FORWARD_INTERVAL_SEC', '10'))}")
                await asyncio.sleep(float(os.environ.get('VALIDATOR_FORWARD_INTERVAL_SEC', '10')))

    def save_state(self):
        super().save_state()
        self.event_provider.save_state()




# The main function parses the configuration and runs the validator.
bt.debug(True)
# bt.trace(True)

if __name__ == "__main__":
    v = Validator(integrations=[
            AzuroProviderIntegration(),
            PolymarketProviderIntegration(),
            AcledProviderIntegration()
        ])
    v.run_in_background_thread()
    time.sleep(2)
    v.thread.join()
