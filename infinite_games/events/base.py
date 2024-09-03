
import asyncio
from collections import defaultdict
import os
import pickle
import sqlite3
import time
import traceback
import bittensor as bt
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Type

from infinite_games.utils.misc import split_chunks


# defines a time window for grouping submissions based on a specified number of minutes
CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES = 60 * 4
CLUSTER_EPOCH_2024 = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, month=1, day=1)


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
    registered_date: datetime
    market_type: str
    description: str
    starts: datetime
    resolve_date: datetime
    answer: Optional[int]
    local_updated_at: datetime
    status: EventStatus
    miner_predictions: Dict[int, Dict[int, Dict[any, any]]]
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

    def latest_submit_date(self, pe: ProviderEvent) -> timedelta:
        return pe.starts or pe.resolve_date

    def log(self, msg):
        bt.logging.debug(f'{self.provider_name().capitalize()}: {msg}')

    def error(self, msg):
        bt.logging.error(f'{self.provider_name()}: {msg}')

    async def get_single_event(self, event_id) -> ProviderEvent:
        pass

    async def sync_events(self, start_from: int = None) -> AsyncIterator[ProviderEvent]:
        pass


class EventAggregator:

    def __init__(self, state_path: str, db_path: str = 'database.db'):

        self.registered_events: Dict[str, ProviderEvent] = {

        }
        self.integrations: Dict[str, ProviderIntegration] = None
        self.state_path = state_path
        # This hook called both when refetching events from provider
        self.event_update_hook_fn: Optional[Callable[[ProviderEvent], None]] = None
        self.WATCH_EVENTS_DELAY = 5
        self.COLLECTOR_WATCH_EVENTS_DELAY = 30
        self.MAX_PROVIDER_CONCURRENT_TASKS = 3
        self.db_path = db_path
        self.create_tables()
        # loop = asyncio.get_event_loop()
        # loop.create_task(self._watch_events())

    @classmethod
    async def create(cls, state_path: str, integrations: List[Type[ProviderIntegration]], db_path='database.db'):
        self = cls(state_path, db_path=db_path)
        self.integrations = {
            integration.provider_name(): await integration._ainit() for integration in integrations
        }

        return self

    def get_registered_event(self, provider_name: str, event_id: str):
        return self.get_event(f'{provider_name}-{event_id}')

    def get_provider_pending_events(self, integration: ProviderIntegration) -> List[ProviderEvent]:
        pe_events = []
        for key, pe in self.registered_events.items():
            if pe.market_type == integration.provider_name() and pe.status == EventStatus.PENDING:
                pe_events.append(pe)
        return pe_events

    async def _sync_provider(self, integration: ProviderIntegration):
        async for event in integration.sync_events():

            self.register_or_update_event(event)

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
            except Exception as e:
                bt.logging.error(e)
                bt.logging.error('Could not pull events.. Retry..')
                print(traceback.format_exc())
            await asyncio.sleep(self.COLLECTOR_WATCH_EVENTS_DELAY)

    async def check_event(self, event_data: ProviderEvent):
        processed_already = event_data.metadata.get('processed', False)
        self.log(f'Update Event {event_data} {event_data.status} {processed_already=} ')

        if event_data.status in [EventStatus.PENDING, EventStatus.SETTLED]:
            integration = self.integrations.get(event_data.market_type)
            if not integration:
                bt.logging.error(f'No integration found for event {event_data.market_type} - {event_data.event_id}')
                return

            try:
                updated_event_data: ProviderEvent = await integration.get_single_event(event_data.event_id)
                if updated_event_data:
                    # self.log(f'Event updated {updated_event_data.event_id}')
                    self.register_or_update_event(updated_event_data)
                    # self.update_event(updated_event_data)
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
        events = self.get_events(statuses=[EventStatus.PENDING, EventStatus.SETTLED], processed=False)
        if events:
            events.sort(key=lambda e: e.resolve_date or e.starts)
            return events[0: n]

    def log_upcoming(self, n):
        self.log(f'*** (Upcoming / In Progress) {n} events ***')
        sooner_events = self.get_upcoming_events(n)
        if sooner_events:
            for i, event in enumerate(sooner_events):
                time_msg = f'resolve: {event.resolve_date}' if event.resolve_date else f'starts: {event.starts}'
                self.log(f'#{i + 1} : {event.description[:100]}  {time_msg} status: {event.status} {event.event_id}')

    def log_submission_status(self, n):
        self.log(f'*** (Submissions) {n} events ***')
        sooner_events = self.get_upcoming_events(n)
        if sooner_events:
            for i, event in enumerate(sooner_events):
                miner_uids = list(event.miner_predictions.keys())
                self.log(f'#{i + 1} : {event.description[:100]} submissions: {len(miner_uids)} {miner_uids}')

    async def watch_events(self):
        """In base implementation we try to update/check each registered event via get_single_event"""
        self.log("Start watcher...")
        while True:
            # settled events has to be processed/scored, thus watch them too to force process
            pending_events = self.get_events(statuses=[EventStatus.PENDING, EventStatus.SETTLED], processed=False)
            self.log(f'Update events: {len(pending_events)}')
            # self.log(f'Watching: {len(self.registered_events.items())} events')
            # self.log_upcoming(50)
            if len(pending_events) != 0:

                try:
                    events_chunks = split_chunks(list(pending_events), self.MAX_PROVIDER_CONCURRENT_TASKS)
                    async for events in events_chunks:
                        await asyncio.gather(*[self.check_event(event_data) for event_data in events])
                        await asyncio.sleep(self.WATCH_EVENTS_DELAY)
                        self.log(f'Updating events..')
                except Exception as e:
                    self.error("Failed to get event")
                    self.error(e)
                    print(traceback.format_exc())

            self.log(f'Watching: {len(pending_events)} events')
            self.log_upcoming(200)
            # self.log_submission_status(200)
            await asyncio.sleep(2)

    def event_key(self, provider_name, event_id):
        return f'{provider_name}-{event_id}'

    def register_or_update_event(self, pe: ProviderEvent):
        """Adds or updates event. Returns true - if this event not in the list yet"""
        key = self.event_key(pe.market_type, event_id=pe.event_id)
        integration = self.integrations.get(pe.market_type)
        if not integration:
            bt.logging.error(f'No integration found for event {pe.market_type} - {pe.event_id}')
            return
        is_new = self.save_event(pe)
        if not is_new:
            # Naive event update
            if self.event_update_hook_fn and callable(self.event_update_hook_fn):
                try:
                    event: ProviderEvent = self.get_event(key)
                    if event.metadata.get('processed', False) is False and self.event_update_hook_fn(event) is True:
                        self.save_event(pe, True)
                        pass
                    elif event.metadata.get('processed', False) is True:
                        bt.logging.warning(f'Tried to process already processed {event} event!')
                except Exception as e:
                    bt.logging.error(f'Failed to call update hook for event {key}')
                    bt.logging.error(e)
                    bt.logging.error(traceback.format_exc())
                    print(traceback.format_exc())
        else:
            self.log(f'New event:  {key} {pe.description} - {pe.status} ')

        return is_new

    def on_event_updated_hook(self, event_update_hook_fn: Callable[[ProviderEvent], None]):
        """Depending on provider, hook that will be called when we have updates for registered events"""
        self.event_update_hook_fn = event_update_hook_fn

    def get_event(self, event_id):
        """Get single event"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        result = None
        try:
            c = cursor.execute(
                """
                select unique_event_id, event_id, market_type, registered_date, description,starts, resolve_date, outcome,local_updated_at,status, metadata, processed, exported
                from events
                where unique_event_id = ?
                """,
                (event_id,)
            )
            result: List[sqlite3.Row] = c.fetchall()
        except Exception as e:
            bt.logging.error(e)
            bt.logging.error(traceback.format_exc())
        conn.close()
        if result:
            data = dict(result[0])
            pe: ProviderEvent = self.row_to_pe(data)
            integration = self.integrations.get(pe.market_type)
            if not integration:
                bt.logging.warning(f'No integration found for event in database {pe}')
                return None
            return pe
        else:
            return None

    def get_integration(self, pe: ProviderEvent) -> ProviderIntegration:
        integration = self.integrations.get(pe.market_type)
        if not integration:
            bt.logging.error(f'No integration found for event {pe.market_type} - {pe.event_id}')
            return
        return integration
    
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

    def get_events_for_submission(self) -> List[ProviderEvent]:
        """Get events that are available for submission"""
        events = []
        for pe in self.get_events(statuses=[EventStatus.PENDING], processed=False):
            integration = self.integrations.get(pe.market_type)
            if not integration:
                bt.logging.warning(f'No integration found for event {pe}')
                continue
            if integration.available_for_submission(pe):
                events.append(pe)
        return events

    def row_to_pe(self, row) -> ProviderEvent:
        return ProviderEvent(
            row.get('event_id'),
            datetime.fromisoformat(row.get('registered_date')),
            row.get('market_type'),
            row.get('description'),
            datetime.fromisoformat(row.get('starts')) if row.get('starts') else None,
            datetime.fromisoformat(row.get('resolve_date')) if row.get('resolve_date') else None,
            float(row.get('outcome')) if row.get('outcome') else None,
            datetime.fromisoformat(row.get('local_updated_at')) if row.get('local_updated_at') else None,
            int(row.get('status')),
            {},
            {**json.loads(row.get('metadata', '{}')), **{'processed': row.get('processed') == 1}},
        )

    def get_events(self, statuses: List[int]=None, processed=None) -> Iterator[ProviderEvent]:
        """Get all events"""
        if not statuses:
            statuses = (str(EventStatus.PENDING), str(EventStatus.SETTLED), str(EventStatus.DISCARDED))
        else:
            statuses = [str(status) for status in statuses]
        # bt.logging.debug(f'STATUS: {statuses}')
        events = []
        result = []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            if processed is None:
                c = cursor.execute(
                    """
                    select unique_event_id, event_id, market_type, registered_date, description, starts, resolve_date, outcome,local_updated_at,status, metadata, exported
                    from events
                    where status in ({})
                    """.format(','.join(statuses))
                )
            else:
                c = cursor.execute(
                    """
                    select unique_event_id, event_id, market_type, registered_date, description, starts, resolve_date, outcome,local_updated_at,status, metadata, exported
                    from events
                    where status in ({}) and processed = ?
                    """.format(','.join(statuses)),
                    (processed,)
                )
            result: List[sqlite3.Row] = c.fetchall()
        except Exception as e:
            bt.logging.error(e)
            bt.logging.error(traceback.format_exc())
        finally:
            conn.close()
        for row in result:
            data = dict(row)
            pe: ProviderEvent = self.row_to_pe(data)
            integration = self.integrations.get(pe.market_type)
            if not integration:
                bt.logging.warning(f'No integration found for event {pe}')
                continue
            events.append(pe)
        return events

    def load_state(self):
        self.log(f'** Loading events from disk  {self.state_path} **')
        try:
            with open(self.state_path, 'rb') as f:
                try:
                    self.registered_events = pickle.load(f)
                except EOFError as eof:
                    self.error(eof)
                    self.error('**** Could not load events! Your events file is corrupted, deleting it')
                    os.remove(self.state_path)
                    self.registered_events = {}
            bt.logging.debug(f'****** Loaded state from disk - events: {len(self.registered_events.keys())} ******')
        except FileNotFoundError:
            bt.logging.debug("No file found, initialize empty state")
            self.registered_events = {}
        except pickle.UnpicklingError:
            bt.logging.error("Invalid events state, initialize empty state!")
            self.registered_events = {}

    async def get_miner_prediction(self, uid: int, event_id: str) -> Optional[Submission]:
        submission = self.registered_events.get(self.event_key(event_id), {}).get(uid)
        return submission

    def _interval_aggregate_function(self, interval_submissions: List[Submission]):
        avg = sum(submissions.answer for submissions in interval_submissions) / len(interval_submissions)
        return avg

    def _resolve_previous_intervals(self, pe: ProviderEvent, uid: int, last_interval_start_minutes: int) -> bool:
        intervals = pe.miner_predictions.get(uid)
        if not intervals:
            return

        for interval_start_minutes, interval_data in intervals.items():
            total = interval_data.get('total_score')
            if (last_interval_start_minutes is None or interval_start_minutes < last_interval_start_minutes) and total is None:
                interval_data['total_score'] = self._interval_aggregate_function(interval_data['entries'] or [])
        return True

    def create_tables(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # create table if it doesn't exist
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                unique_event_id TEXT,
                minerHotkey TEXT,
                minerUid TEXT,
                predictedOutcome TEXT,
                canOverwrite BOOLEAN,
                outcome TEXT,
                interval_start_minutes INTEGER,
                interval_agg_prediction REAL,
                interval_count INTEGER,
                submitted DATETIME,
                blocktime INTEGER,
                exported INTEGER DEFAULT 0,
                PRIMARY KEY (unique_event_id, interval_start_minutes, minerUid)
            );
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                unique_event_id PRIMARY KEY,
                event_id TEXT,
                market_type TEXT,
                registered_date DATETIME,
                description TEXT,
                starts DATETIME,
                resolve_date DATETIME,
                outcome TEXT,
                local_updated_at DATETIME,
                status TEXT,
                metadata TEXT,
                processed BOOLEAN DEFAULT false,
                exported INTEGER DEFAULT 0
            );

        """
        )

        conn.commit()
        conn.close()

    def migrate_pickle_to_sql(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        events = self.get_events()
        if events and len(events) > 0:
            bt.logging.info('Already migrated, found events in database.')
            cursor.close()
            conn.close()
            return
        else:
            bt.logging.info('Migrating pickle to database...')
        try:

            for pe in list(self.registered_events.values()):
                bt.logging.info(f'Saving {pe} event and submissions.')
                c = cursor.execute(
                    """
                    INSERT into events ( unique_event_id, event_id, market_type, registered_date, description,starts, resolve_date, outcome,local_updated_at,status, metadata)
                    Values (?, ?, ?, ?, ?, ?, ?, ?, ?, ? , ?)
                    ON CONFLICT(unique_event_id)
                    DO NOTHING
                    """,
                    (f'{pe.market_type}-{pe.event_id}', pe.event_id,  pe.market_type, pe.registered_date,  pe.description, pe.starts, pe.resolve_date , pe.answer,datetime.now(tz=timezone.utc), pe.status, json.dumps(pe.metadata)),
                )
                for uid, intervals_dict in pe.miner_predictions.items():
                    for interval_start_minutes, data in intervals_dict.items():
                        agg_prediction = data.get('total_score', 0)
                        total_count = data.get('count', 0)
                        cursor.execute(
                            """
                            INSERT into predictions ( unique_event_id, minerHotkey, minerUid, predictedOutcome,interval_start_minutes,interval_agg_prediction,interval_count,submitted,blocktime)
                            Values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(unique_event_id,  interval_start_minutes, minerUid)
                            DO Nothing""",
                            (f'{pe.market_type}-{pe.event_id}', None, uid, None, interval_start_minutes, agg_prediction , total_count,datetime.now(tz=timezone.utc), 0),
                        )
            if self.registered_events:
                conn.execute("COMMIT")
        except Exception as e:
            bt.logging.error('Error migrating data! Ples')
            bt.logging.error(traceback.format_exc())
            exit()
        conn.close()
        bt.logging.info('Data migrated successfully!')

    def save_event(self, pe: ProviderEvent, processed=False) -> bool:
        """Returns true if new event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        result = []
        tries = 4
        tried = 0
        while tried < tries:
            # bt.logging.info(f'Now time: {datetime.now(tz=timezone.utc)}, {pe} ')
            try:
                c = cursor.execute(
                    """
                    INSERT into events ( unique_event_id, event_id, market_type, registered_date, description,starts, resolve_date, outcome,local_updated_at,status, metadata)
                    Values (?, ?, ?, ?, ?, ?, ?, ?, ?, ? , ?)
                    ON CONFLICT(unique_event_id)
                    DO UPDATE set outcome = ?, status = ?, local_updated_at = ?, processed = ?
                    RETURNING unique_event_id, registered_date, local_updated_at
                    """,
                    (self.event_key(pe.market_type, event_id=pe.event_id), pe.event_id,  pe.market_type, pe.registered_date,  pe.description, pe.starts, pe.resolve_date , pe.answer,pe.registered_date, pe.status, json.dumps(pe.metadata),
                    pe.answer, pe.status, datetime.now(tz=timezone.utc), processed),
                )
                result = c.fetchall()
                # bt.logging.debug(result)
                conn.execute("COMMIT")
                break
            except Exception as e:
                if 'locked' in str(e):
                    bt.logging.warning(
                        f"Database locked, retry {tried + 1}.."
                    )
                    time.sleep(1 + (2 * tried))

                else:
                    bt.logging.error(e)
                    bt.logging.error(traceback.format_exc())
                    break
            tried += 1

        conn.close()
        if result and result[0]:
            # bt.logging.debug(result)
            return result[0][1] == result[0][2]
        return False

    def update_cluster_prediction(self, pe: ProviderEvent, uid: int, blocktime: int, interval_start_minutes: int, new_prediction):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        tries = 4
        tried = 0
        while tried < tries:

            try:
                cursor.execute(
                    """
                    INSERT into predictions ( unique_event_id, minerHotkey, minerUid, predictedOutcome,interval_start_minutes,interval_agg_prediction,interval_count,submitted,blocktime)
                    Values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(unique_event_id,  interval_start_minutes, minerUid)
                    DO UPDATE set interval_agg_prediction = (interval_agg_prediction * interval_count + ?) / (interval_count + 1), interval_count = interval_count + 1""",
                    (self.event_key(pe.market_type, event_id=pe.event_id), None, uid, None, interval_start_minutes, new_prediction , 1,datetime.now(tz=timezone.utc), blocktime, new_prediction),
                )
                conn.execute("COMMIT")
                break
            except sqlite3.OperationalError as e:
                if 'locked' in str(e):
                    bt.logging.warning(
                        f"Database locked, retry {tried + 1}.."
                    )
                    time.sleep(1 + (2 * tried))
                    # tried += 1
                else:
                    bt.logging.error(traceback.format_exc())
                    bt.logging.error(
                        f"Error setting miner predictions {uid=} {pe} {e} "
                    )
                    break
            except Exception as e:
                bt.logging.info(e)
                bt.logging.info(traceback.format_exc())
                break
            tried += 1
        conn.close()

    def get_event_predictions(self, pe: ProviderEvent):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        result = []
        try:
            c = cursor.execute(
                """
                select unique_event_id, minerHotkey, minerUid, predictedOutcome,interval_start_minutes,interval_agg_prediction,interval_count,submitted,blocktime
                from predictions
                where unique_event_id = ?
                """,
                (self.event_key(pe.market_type, event_id=pe.event_id),)
            )
            result: List[sqlite3.Row] = c.fetchall()
        except Exception as e:
            bt.logging.error(e)
            bt.logging.error(traceback.format_exc())
        conn.close()
        output = defaultdict(dict)
        for row in result:
            interval_prediction = dict(row)
            if int(interval_prediction['interval_start_minutes']) not in output[int(interval_prediction['minerUid'])]:
                output[int(interval_prediction['minerUid'])][int(interval_prediction['interval_start_minutes'])] = interval_prediction
        return output

    async def miner_predict(self, pe: ProviderEvent, uid: int, answer: float, interval_start_minutes: int, blocktime: int) -> Submission:
        # bt.logging.info(f'{uid=} retrieving submission..')
        submission: Submission = pe.miner_predictions.get(uid)
        if pe.market_type == 'azuro':
            self.update_cluster_prediction(pe, uid, blocktime, 0, answer)
        else:
            # aggregate all previous intervals if not yet
            # self._resolve_previous_intervals(pe, uid, interval_start_minutes)
            # bt.logging.info(f"{uid=} identifying interval for {interval_start_minutes=} {pe}")
            self.update_cluster_prediction(pe, uid, blocktime, interval_start_minutes, answer)

        return submission
