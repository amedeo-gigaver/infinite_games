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
import logging
import os
import time
import traceback
os.environ['USE_TORCH'] = '1'
import bittensor as bt
import torch
import infinite_games

# import base validator class which takes care of most of the boilerplate
from infinite_games.base.validator import BaseValidatorNeuron
from infinite_games.events.base import EventAggregator, EventStatus, ProviderEvent, Submission
from infinite_games.events.azuro import AzuroProviderIntegration
from infinite_games.events.polymarket import PolymarketProviderIntegration


class Validator(BaseValidatorNeuron):
    """
    Validator neuron class.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.active_markets = {}
        self.blocktime = 0
        self.event_provider = None
        self.SEND_LOGS_INTERVAL = 60 * 60

    async def initialize_provider(self):
        if not self.event_provider:
            self.event_provider: EventAggregator = await EventAggregator.create(
                state_path=self.config.neuron.full_path + '/events.pickle',
                integrations=[
                    AzuroProviderIntegration(max_pending_events=6),
                    PolymarketProviderIntegration()
                ]
            )
            self.event_provider.load_state()
            self.event_provider.on_event_updated_hook(self.on_event_update)
            # watch for existing registered events
            self.loop.create_task(self.event_provider.watch_events())
            # pull new markets
            self.loop.create_task(self.event_provider.collect_events())
            bt.logging.info(f'TARGET_MONITOR_HOTKEY: {os.environ.get("TARGET_MONITOR_HOTKEY", "None")}')
            bt.logging.info(f'GRAFANA_API_KEY: {os.environ.get("GRAFANA_API_KEY", "None")}')
            if self.wallet.hotkey.ss58_address == os.environ.get('TARGET_MONITOR_HOTKEY'):
                self.loop.create_task(self.send_stats())
            bt.logging.debug("Provider initialized..")

    async def send_stats(self):
        bt.logging.info('Scheduling sending average stats.')
        while True:

            all_uids = [uid for uid in range(self.metagraph.n.item())]
            bt.logging.debug(f"Sending daily average total: {self.average_scores}")
            self.send_average_scores(miner_scores=list(zip(all_uids, self.average_scores.tolist(), self.scores.tolist())))
            await asyncio.sleep(self.SEND_LOGS_INTERVAL)

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
            scores = []
            if not predictions:
                bt.logging.warning(f"No predictions for {pe} skipping..")
                return True

            bt.logging.info(f'Miners to update: {len(miner_uids)} submissions: {len(predictions.keys())} from {self.metagraph.n.item()}')
            for uid in miner_uids:
                submission: Submission = predictions.get(uid.item())
                ans = None
                if submission:
                    ans = submission.answer
                # bt.logging.debug(f'Submission of {uid=} {ans=}')
                if ans is None:
                    scores.append(0)
                else:
                    ans = max(0, min(1, ans))  # Clamp the answer
                    scores.append(1 - ((ans - correct_ans)**2))
            self.update_scores(torch.FloatTensor(scores), miner_uids)
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
        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            # Pass the synapse to the miner.
            synapse=synapse,
            # Do not deserialize the response so that we have access to the raw response.
            deserialize=False,
        )

        # synapse.events['azuro-0x7f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927405030000000000000357953524_142'] = {
        #     'event_id': '0x7f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927405030000000000000357953524_142',
        #     'probability': 0.7,
        #     'market_type': 'azuro'
        # }

        # Update answers
        miners_activity = set()
        for (uid, resp) in zip(miner_uids, responses):
            miner_submitted = set()
            for (event_id, event_data) in resp.events.items():
                market_event_id = event_data.get('event_id')
                provider_name = event_data.get('market_type')
                score = event_data.get('probability')

                provider_event = self.event_provider.get_registered_event(provider_name, market_event_id)
                # bt.logging.debug(f'Got miner submission {uid=} {event_id=} {score is None} {provider_event}')
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
                    miners_activity.add(uid)
                    miner_submitted.add(event_id)
                    self.event_provider.miner_predict(provider_event, uid.item(), score, self.block)
                else:
                    # bt.logging.warning(f'Submission received, but this event is not open for submissions miner {uid=} {event_id=} {score=}')
                    continue
            # if len(miner_submitted) > 0:

            # bt.logging.info(f'uid: {uid.item()} got prediction for events: {len(miner_submitted)}')

        if miners_activity:
            bt.logging.info("Processed miner responses.")
        else:
            bt.logging.info('No miner submissions received')
        self.blocktime += 1
        while block_start == self.block:
            await asyncio.sleep(10)

    def save_state(self):
        super().save_state()
        self.event_provider.save_state()




# The main function parses the configuration and runs the validator.
bt.debug(True)
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            validator.print_info()
            bt.logging.info("Validator running...", time.time())
            time.sleep(20)
