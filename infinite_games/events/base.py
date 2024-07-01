
import asyncio
import pickle
import traceback
import bittensor as bt
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type

from infinite_games.utils.misc import split_chunks


@dataclass
class Submission:
    """Miner submission data"""

    submitted_ts: int
    # this is blockchain block time when we send this submission.
    blocktime: int
    answer: float


class EventStatus:
    """Generic event status"""
    DISCARDED = 1
    PENDING = 2
    SETTLED = 3
    # In case of errors
    NOT_IMPLEMENTED = 4


@dataclass
class ProviderEvent:
    event_id: str
    market_type: str
    description: str
    starts: datetime
    resolve_date: datetime
    answer: Optional[int]
    local_updated_at: datetime
    status: EventStatus
    miner_predictions: Dict[int, Submission]
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return f'{self.market_type} {self.event_id}'


class ProviderIntegration:

    def __init__(self, max_pending_events=None):
        self.max_pending_events = max_pending_events
        if self.max_pending_events:
            self.log(f'Registered pending event limit set to {self.max_pending_events}')

    async def _ainit(self):
        """This method is for async initializations of providers"""
        pass

    @classmethod
    def provider_name(cls) -> str:
        pass

    def available_for_submission(self, pe: ProviderEvent) -> bool:
        return True

    def log(self, msg):
        bt.logging.debug(f'{self.provider_name().capitalize()}: {msg}')

    def error(self, msg):
        bt.logging.error(f'{self.provider_name()}: {msg}')

    async def get_single_event(self, event_id) -> ProviderEvent:
        pass

    async def sync_events(self, start_from: int = None) -> AsyncIterator[ProviderEvent]:
        pass


class EventAggregator:

    def __init__(self, state_path: str):
        self.registered_events: Dict[str, ProviderEvent] = {

        }

        self.registered_events: Dict[str, ProviderEvent] = {

        }
        self.integrations: Dict[str, ProviderIntegration] = None
        self.state_path = state_path
        # This hook called both when refetching events from provider
        self.event_update_hook_fn: Optional[Callable[[ProviderEvent], None]] = None
        self.WATCH_EVENTS_DELAY = 5
        self.COLLECTOR_WATCH_EVENTS_DELAY = 30
        self.MAX_PROVIDER_CONCURRENT_TASKS = 5
        # loop = asyncio.get_event_loop()
        # loop.create_task(self._watch_events())

    @classmethod
    async def create(cls, state_path: str, integrations: List[Type[ProviderIntegration]]):
        self = cls(state_path)
        self.integrations = {
            integration.provider_name(): await integration._ainit() for integration in integrations
        }

        return self

    def get_registered_event(self, provider_name: str, event_id: str):
        return self.registered_events.get(self.event_key(provider_name, event_id))

    def get_provider_pending_events(self, integration: ProviderIntegration) -> List[ProviderEvent]:
        pe_events = []
        for key, pe in self.registered_events.items():
            if pe.market_type == integration.provider_name() and pe.status == EventStatus.PENDING:
                pe_events.append(pe)
        return pe_events

    async def _sync_provider(self, integration: ProviderIntegration):
        async for event in integration.sync_events():

            self.register_event(event)

    async def collect_events(self):
        if not self.integrations:
            self.error('Please add integration to provider and restart your script.')
            raise Exception("No Provider Integrations Found. Please Add 'ProviderIntegration' compatible integrations ")
        self.log('Start collector..')
        while True:
            self.log(f'Pulling events from providers. Current: {len(self.registered_events.items())}')
            try:
                tasks = [self._sync_provider(integration) for _, integration in self.integrations.items()]
                await asyncio.gather(*tasks)
            except Exception:
                bt.logging.error('Could not pull events.. Retry..')
                print(traceback.format_exc())
            await asyncio.sleep(self.COLLECTOR_WATCH_EVENTS_DELAY)

    async def check_event(self, event_data: ProviderEvent):
        # self.log(f'Update Event {event_data.event_id}')
        if event_data.status in [EventStatus.PENDING, EventStatus.SETTLED]:
            integration = self.integrations.get(event_data.market_type)
            if not integration:
                bt.logging.error(f'No integration found for event {event_data.market_type} - {event_data.event_id}')
                return

            try:
                updated_event_data: ProviderEvent = await integration.get_single_event(event_data.event_id)
                if updated_event_data:
                    # self.log(f'Event updated {updated_event_data.event_id}')
                    self.update_event(updated_event_data)
                else:
                    self.warning(f'Could not update event {event_data}')

            except Exception as e:
                bt.logging.error(f'Failed to check event {event_data}')
                bt.logging.error(e)
                print(traceback.format_exc())
            # bt.logging.debug(f'Fetching done {event_id}')
            # await asyncio.sleep(2)
  
    def log(self, msg):
        bt.logging.info(f'{self.__class__.__name__} {msg}')

    def error(self, msg):
        bt.logging.error(f'{self.__class__.__name__} {msg}')

    def warning(self, msg):
        bt.logging.warning(f'{self.__class__.__name__} {msg}')

    def debug(self, msg):
        bt.logging.debug(f'{self.__class__.__name__} {msg}')

    def get_upcoming_events(self, n):
        if self.registered_events:
            events = list(self.registered_events.values())
            events.sort(key=lambda e: e.resolve_date or e.starts)
            return events[0: n]

    def log_upcoming(self, n):
        self.log(f'*** (Upcoming / In Progress) {n} events ***')
        sooner_events = self.get_upcoming_events(n)
        if sooner_events:
            for i, event in enumerate(sooner_events):
                time_msg = f'resolve: {event.resolve_date}' if event.resolve_date else f'starts: {event.starts}'
                self.log(f'#{i + 1} : {event.description[:100]}  {time_msg} status: {event.status} {event.event_id}')

    async def watch_events(self):
        """In base implementation we try to update/check each registered event via get_single_event"""
        self.log("Start watcher...")
        while True:
            # self.collect_events()
            self.log(f'Update events: {len(self.registered_events.items())}')
            # self.log(f'Watching: {len(self.registered_events.items())} events')
            # self.log_upcoming(50)
            try:
                events_chunks = split_chunks(list(self.registered_events.items()), self.MAX_PROVIDER_CONCURRENT_TASKS)
                async for events in events_chunks:
                    # self.log(f'Checking next {len(events)} events..')
                    await asyncio.gather(*[self.check_event(event_data) for _, event_data in events])
                    await asyncio.sleep(self.WATCH_EVENTS_DELAY)
            except Exception as e:
                self.error("Failed to get event")
                self.error(e)
                print(traceback.format_exc())

            self.log(f'Watching: {len(self.registered_events.items())} events')
            self.log_upcoming(50)

    def event_key(self, provider_name, event_id):
        return f'{provider_name}-{event_id}'

    def register_event(self, pe: ProviderEvent):
        """Adds or updates event. Returns true - if this event not in the list yet"""
        key = self.event_key(pe.market_type, event_id=pe.event_id)
        if self.registered_events.get(key):
            # Naive event update
            self.update_event(pe)
        else:
            integration = self.integrations.get(pe.market_type)
            if not integration:
                bt.logging.error(f'No integration found for event {pe.market_type} - {pe.event_id}')
                return
            if integration.max_pending_events and len(self.get_provider_pending_events(integration)) >= integration.max_pending_events:
                return
            self.log(f'New event:  {key} {pe.description} - {pe.status} ')
            self.registered_events[key] = pe

        return True

    def update_event(self, pe: ProviderEvent):
        """Updates event"""
        key = self.event_key(provider_name=pe.market_type, event_id=pe.event_id)
        if not self.registered_events.get(self.event_key(provider_name=pe.market_type, event_id=pe.event_id)):
            bt.logging.error(f'No event found in registry {pe.market_type} {pe.event_id}!')
            return False
        # self.log(f'Updating event: {description}, {start_time} status: {event_status}', )
        self.registered_events[key] = ProviderEvent(
            pe.event_id,
            pe.market_type,
            pe.description,
            pe.starts,
            pe.resolve_date,
            pe.answer,
            pe.local_updated_at,
            pe.status,
            self.registered_events[key].miner_predictions,
            # {
            #     0: Submission(datetime.now().timestamp(), 1)
            # },
            pe.metadata,
        )
        if self.event_update_hook_fn and callable(self.event_update_hook_fn):
            try:
                if self.event_update_hook_fn(self.registered_events.get(key)) is True:
                    del self.registered_events[key]
            except Exception as e:
                bt.logging.error(f'Failed to call update hook for event {key}')
                bt.logging.error(e)
                print(traceback.format_exc())

        return True

    def on_event_updated_hook(self, event_update_hook_fn: Callable[[ProviderEvent], None]):
        """Depending on provider, hook that will be called when we have updates for registered events"""
        self.event_update_hook_fn = event_update_hook_fn

    def get_event(self, event_id):
        return self.registered_events.get(self.get_event_key(event_id))

    def remove_event(self, pe: ProviderEvent) -> bool:
        """Removed event"""
        key = self.event_key(provider_name=pe.market_type, event_id=pe.event_id)
        if key in self.registered_events:

            del self.registered_events[key]
            return True
        return False

    def save_state(self):
        with open(self.state_path, 'wb') as f:
            pickle.dump(self.registered_events, f)

    def get_events_for_submission(self) -> AsyncIterator[ProviderEvent]:
        """Get events that are available for submission"""
        events = []
        for _, pe in self.registered_events.items():
            integration = self.integrations.get(pe.market_type)
            if not integration:
                bt.logging.warning(f'No integration found for event {pe}')
                continue
            if integration.available_for_submission(pe):
                events.append(pe)
        return events

    def load_state(self):
        self.log('** Loading events from disk **')
        try:
            with open(self.state_path, 'rb') as f:
                self.registered_events = pickle.load(f)
            bt.logging.debug(f'****** Loaded state from disk - events: {len(self.registered_events.keys())} ******')
        except FileNotFoundError:
            bt.logging.debug("No file found, initialize empty state")
            self.registered_events = {}
        except pickle.UnpicklingError:
            bt.logging.error("Invalid events state, initialize empty state!")

    async def get_miner_prediction(self, uid: int, event_id: str) -> Optional[Submission]:
        submission = self.registered_events.get(self.event_key(event_id), {}).get(uid)
        return submission

    def miner_predict(self, pe: ProviderEvent, uid: int, answer: float, blocktime: int) -> Submission:
        submission: Submission = pe.miner_predictions.get(uid)

        if submission:
            if abs(submission.answer - answer) > 0.01:
                bt.logging.info(f'Overriding miner {uid=} submission for {pe.event_id=} from {submission.answer} to {answer}')
        new_submission = Submission(
            submitted_ts=datetime.now().timestamp(),
            answer=answer,
            blocktime=blocktime
        )
        pe.miner_predictions[uid] = new_submission
        return submission
