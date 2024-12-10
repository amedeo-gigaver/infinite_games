import asyncio
import json
import os
import shutil
import sqlite3
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Type

import backoff
import bittensor as bt
import pandas as pd
from bittensor.chain_data import AxonInfo

from infinite_games.utils.misc import split_chunks
from infinite_games.utils.uids import miner_count_in_db

# defines a time window for grouping submissions based on a specified number of minutes
CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES = 60 * 4
CLUSTER_EPOCH_2024 = datetime.now(timezone.utc).replace(
    hour=0, minute=0, second=0, microsecond=0, month=1, day=1
)


class EventRemovedException(Exception):
    pass


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
        return f"{self.market_type} {self.event_id}"

    def __repr__(self) -> str:
        return (
            f"{self.market_type=} {self.event_id=} {self.registered_date=} "
            f"{self.description=} {self.starts=} {self.resolve_date=} "
            f"{self.answer=} {self.local_updated_at=} {self.status=} "
            f"{json.dumps(self.metadata)}"
        )


class ProviderIntegration:
    def __init__(self, max_pending_events=None):
        self.max_pending_events = max_pending_events
        if self.max_pending_events:
            self.log(f"Registered pending event limit set to {self.max_pending_events}")

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
        bt.logging.debug(f"{self.provider_name().capitalize()}: {msg}")

    def error(self, msg):
        bt.logging.error(f"{self.provider_name()}: {msg}")

    async def get_single_event(self, event_id) -> ProviderEvent:
        pass

    async def sync_events(self, start_from: int = None) -> AsyncIterator[ProviderEvent]:
        pass


class EventAggregator:
    def __init__(self, state_path: str, db_path: str = "database.db"):
        self.registered_events: Dict[str, ProviderEvent] = {}
        self.integrations: Dict[str, ProviderIntegration] = None
        self.state_path = state_path
        # This hook called both when refetching events from provider
        self.event_update_hook_fn: Optional[Callable[[ProviderEvent], None]] = None
        self.WATCH_EVENTS_DELAY = 5
        self.COLLECTOR_WATCH_EVENTS_DELAY = 30
        self.MAX_PROVIDER_CONCURRENT_TASKS = 3
        self.db_path = db_path
        self.init_migrations()
        # loop = asyncio.get_event_loop()
        # loop.create_task(self._watch_events())

    @classmethod
    async def create(
        cls, state_path: str, integrations: List[Type[ProviderIntegration]], db_path="database.db"
    ):
        self = cls(state_path, db_path=db_path)
        self.integrations = {
            integration.provider_name(): await integration._ainit() for integration in integrations
        }

        return self

    def get_registered_event(self, unique_event_id: str):
        return self.get_event(unique_event_id)

    async def _sync_provider(self, integration: ProviderIntegration):
        async for event in integration.sync_events():
            self.register_or_update_event(event)

    async def collect_events(self):
        if not self.integrations:
            self.error("Please add integration to provider and restart your script.")
            raise Exception(
                "No Provider Integrations Found. Please Add 'ProviderIntegration' compatible integrations "
            )
        self.log("Start collector..")
        while True:
            pending_events = self.get_events(
                statuses=[EventStatus.PENDING, EventStatus.SETTLED], processed=False
            )
            self.log(f"Pulling events from providers. Current: {len(pending_events)}")
            try:
                tasks = [
                    self._sync_provider(integration) for _, integration in self.integrations.items()
                ]
                await asyncio.gather(*tasks)
            except Exception as e:
                bt.logging.error(f"Could not pull events.. Retry.. Exception: {repr(e)}")
                bt.logging.error(traceback.format_exc())
            await asyncio.sleep(self.COLLECTOR_WATCH_EVENTS_DELAY)

    async def check_event(self, event_data: ProviderEvent):
        processed_already = event_data.metadata.get("processed", False)
        market_type = event_data.metadata.get("market_type", event_data.market_type)
        event_text = f"{market_type} {event_data.event_id}"
        self.log(f"Update Event {event_text} {event_data.status} {processed_already=} ")

        if event_data.status in [EventStatus.PENDING, EventStatus.SETTLED]:
            integration = self.integrations.get(event_data.market_type)
            if not integration:
                bt.logging.error(
                    f"No integration found for event {event_data.market_type} - {event_data.event_id}"
                )
                return

            try:
                updated_event_data: ProviderEvent = await integration.get_single_event(
                    event_data.event_id
                )
                if updated_event_data:
                    # self.log(f'Event updated {updated_event_data.event_id}')
                    self.register_or_update_event(updated_event_data)
                    # self.update_event(updated_event_data)
                else:
                    self.warning(f"Could not update event {event_data}")
            except EventRemovedException:
                self.remove_event(event_data)
            except Exception as e:
                bt.logging.error(f"Failed to check event {event_data}: {repr(e)}")
                bt.logging.error(traceback.format_exc())

    def log(self, msg):
        bt.logging.info(f"{self.__class__.__name__} {msg}")

    def error(self, msg):
        bt.logging.error(f"{self.__class__.__name__} {msg}")

    def warning(self, msg):
        bt.logging.warning(f"{self.__class__.__name__} {msg}")

    def debug(self, msg):
        bt.logging.debug(f"{self.__class__.__name__} {msg}")

    def get_upcoming_events(self, n):
        events = self.get_events(
            statuses=[EventStatus.PENDING, EventStatus.SETTLED], processed=False
        )
        if events:
            events.sort(key=lambda e: e.resolve_date or e.starts)
            return events[0:n]

    def log_upcoming(self, n):
        self.log(f"*** (Upcoming / In Progress) {n} events ***")
        sooner_events = self.get_upcoming_events(n)
        if sooner_events:
            for i, event in enumerate(sooner_events):
                time_msg = (
                    f"resolve: {event.resolve_date}"
                    if event.resolve_date
                    else f"starts: {event.starts}"
                )
                self.log(
                    f"#{i + 1} : {event.description[:100]}  {time_msg} status: {event.status} {event.event_id}"
                )

    def log_submission_status(self, n):
        self.log(f"*** (Submissions) {n} events ***")
        sooner_events = self.get_upcoming_events(n)
        if sooner_events:
            for i, event in enumerate(sooner_events):
                miner_uids = list(event.miner_predictions.keys())
                self.log(
                    f"#{i + 1} : {event.description[:100]} submissions: {len(miner_uids)} {miner_uids}"
                )

    async def watch_events(self):
        """In base implementation we try to update/check each registered event via get_single_event"""
        self.log("Start watcher...")
        while True:
            # settled events has to be processed/scored, thus watch them too to force process
            pending_events = self.get_events(
                statuses=[EventStatus.PENDING, EventStatus.SETTLED], processed=False
            )
            self.log(f"Update events: {len(pending_events)}")
            # self.log_upcoming(50)
            if len(pending_events) != 0:
                try:
                    events_chunks = split_chunks(
                        list(pending_events), self.MAX_PROVIDER_CONCURRENT_TASKS
                    )
                    async for events in events_chunks:
                        await asyncio.gather(
                            *[self.check_event(event_data) for event_data in events]
                        )
                        await asyncio.sleep(self.WATCH_EVENTS_DELAY)
                        self.log("Updating events..")
                except Exception as e:
                    self.error(f"Failed to get event: {repr(e)}")
                    self.error(traceback.format_exc())

            self.log(f"Watching: {len(pending_events)} events")
            self.log_upcoming(200)
            # self.log_submission_status(200)
            await asyncio.sleep(2)

    def event_key(self, provider_name, event_id):
        return f"{provider_name}-{event_id}"

    def register_or_update_event(self, pe: ProviderEvent):
        """Adds or updates event. Returns true - if this event not in the list yet"""
        key = self.event_key(pe.market_type, event_id=pe.event_id)
        integration = self.integrations.get(pe.market_type)
        if not integration:
            bt.logging.error(f"No integration found for event {pe.market_type} - {pe.event_id}")
            return
        is_new = self.save_event(pe)
        if not is_new:
            # Naive event update
            if self.event_update_hook_fn and callable(self.event_update_hook_fn):
                try:
                    event: ProviderEvent = self.get_event(key)
                    if not event:
                        bt.logging.error(f"Could not get updated event from database {repr(pe)}")
                        return
                    if (
                        event.metadata.get("processed", False) is False
                        and self.event_update_hook_fn(event) is True
                    ):
                        self.save_event(pe, True)
                        pass
                    elif event.metadata.get("processed", False) is True:
                        bt.logging.warning(f"Tried to process already processed {event} event!")
                except Exception as e:
                    bt.logging.error(f"Failed to call update hook for event {key}: {repr(e)}")
                    bt.logging.error(traceback.format_exc())
                    print(traceback.format_exc())
        else:
            self.log(f"New event:  {key} {pe.description} - {pe.status} ")

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
                (event_id,),
            )
            result: List[sqlite3.Row] = c.fetchall()
        except Exception as e:
            bt.logging.error(f"Error fetching event {event_id}: {repr(e)}")
            bt.logging.error(traceback.format_exc())
        conn.close()
        if result:
            data = dict(result[0])
            pe: ProviderEvent = self.row_to_pe(data)
            integration = self.integrations.get(pe.market_type)
            if not integration:
                bt.logging.warning(f"No integration found for event in database {repr(pe)}")
                return None
            return pe
        else:
            return None

    def get_integration(self, pe: ProviderEvent) -> ProviderIntegration:
        integration = self.integrations.get(pe.market_type)
        if not integration:
            bt.logging.error(f"No integration found for event {pe.market_type} - {pe.event_id}")
            return
        return integration

    def remove_event(self, pe: ProviderEvent) -> bool:
        """Removed event"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        tries = 4
        tried = 0
        while tried < tries:
            try:
                c.execute(
                    """
                    delete from events
                    where unique_event_id = ?
                    """,
                    (f"{pe.market_type}-{pe.event_id}",),
                )
                bt.logging.info(f"Removed event {repr(pe)}..")
                conn.commit()
                return True
            except Exception as e:
                if "locked" in str(e):
                    bt.logging.warning(f"Database locked, retry {tried + 1}..")
                    time.sleep(1 + (2 * tried))

                else:
                    bt.logging.error(f"Error removing event {repr(pe)}: {repr(e)}")
                    bt.logging.error(traceback.format_exc())
                    break

            tried += 1

        conn.close()
        return False

    def save_state(self):
        pass

    def get_events_for_submission(self) -> List[ProviderEvent]:
        """Get events that are available for submission"""
        events = []
        for pe in self.get_events(statuses=[EventStatus.PENDING], processed=False):
            integration = self.integrations.get(pe.market_type)
            if not integration:
                bt.logging.warning(f"No integration found for event {repr(pe)}")
                continue
            if integration.available_for_submission(pe):
                events.append(pe)
        return events

    def row_to_pe(self, row) -> ProviderEvent:
        return ProviderEvent(
            row.get("event_id"),
            datetime.fromisoformat(row.get("registered_date")),
            row.get("market_type"),
            row.get("description"),
            datetime.fromisoformat(row.get("starts")) if row.get("starts") else None,
            datetime.fromisoformat(row.get("resolve_date")) if row.get("resolve_date") else None,
            float(row.get("outcome")) if row.get("outcome") else None,
            (
                datetime.fromisoformat(row.get("local_updated_at"))
                if row.get("local_updated_at")
                else None
            ),
            int(row.get("status")),
            {},
            {**json.loads(row.get("metadata", "{}")), **{"processed": row.get("processed") == 1}},
        )

    def get_events(self, statuses: List[int] = None, processed=None) -> Iterator[ProviderEvent]:
        """Get all events"""
        if not statuses:
            statuses = (
                str(EventStatus.PENDING),
                str(EventStatus.SETTLED),
                str(EventStatus.DISCARDED),
            )
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
                    """.format(
                        ",".join(statuses)
                    )
                )
            else:
                c = cursor.execute(
                    """
                    select unique_event_id, event_id, market_type, registered_date, description, starts, resolve_date, outcome,local_updated_at,status, metadata, exported
                    from events
                    where status in ({}) and processed = ?
                    """.format(
                        ",".join(statuses)
                    ),
                    (processed,),
                )
            result: List[sqlite3.Row] = c.fetchall()
        except Exception as e:
            bt.logging.error(f"Error fetching events: {repr(e)}")
            bt.logging.error(traceback.format_exc())
        finally:
            conn.close()
        for row in result:
            data = dict(row)
            pe: ProviderEvent = self.row_to_pe(data)
            events.append(pe)
        return events

    def load_state(self):
        pass

    def _interval_aggregate_function(self, interval_submissions: List[Submission]):
        avg = sum(submissions.answer for submissions in interval_submissions) / len(
            interval_submissions
        )
        return avg

    def _resolve_previous_intervals(
        self, pe: ProviderEvent, uid: int, last_interval_start_minutes: int
    ) -> bool:
        intervals = pe.miner_predictions.get(uid)
        if not intervals:
            return

        for interval_start_minutes, interval_data in intervals.items():
            total = interval_data.get("total_score")
            if (
                last_interval_start_minutes is None
                or interval_start_minutes < last_interval_start_minutes
            ) and total is None:
                interval_data["total_score"] = self._interval_aggregate_function(
                    interval_data["entries"] or []
                )
        return True

    def init_migrations(self):
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

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS miners (
                miner_hotkey TEXT,
                miner_uid TEXT,
                node_ip TEXT,
                registered_date DATETIME,
                last_updated DATETIME,
                blocktime INTEGER,
                blocklisted boolean DEFAULT false,
                PRIMARY KEY (miner_hotkey, miner_uid)
            );

        """
        )
        tries = 4
        tried = 0
        bt.logging.info("Migrate providers to ifgames..")
        try:
            current_dir = os.getcwd()
            total, used, free = shutil.disk_usage(current_dir)
        except Exception as e:
            self.error(f"Error checking disk space: {repr(e)}")
            self.error(traceback.format_exc())
            self.error("Error checking disk space, continue for migration..")
        else:
            free_gb = free / (1024**3)
            if os.environ.get("ENV") != "pytest" and free_gb < 20:
                self.error(
                    f"Not enough disk space ❌. Only {free_gb:.2f} GB available. Please make sure that you have available space then restart the process."
                )
                exit(1)
            self.log(f"Free space: {free_gb:.2f} GB")
        while tried < tries:
            try:
                result = c.execute(
                    """
                    select true from events where market_type='ifgames' limit 1
                    """
                )

                if len(result.fetchall()) > 0:
                    self.log("Already migrated to ifgames skip...")
                    break
                c.execute(
                    """
                    delete from events where market_type='azuro' and status in (2, 3) and processed = false
                    """
                )
                result = c.execute(
                    """
                    select unique_event_id from events where status in (2, 3) and processed = false
                    """
                )

                unique_event_ids = [event_id[0] for event_id in result.fetchall()]
                for event in self.get_events(
                    statuses=[EventStatus.PENDING, EventStatus.SETTLED], processed=False
                ):
                    if event.market_type == "polymarket":
                        print(f"Migrating {event}..")
                        event.metadata["market_type"] = "polymarket"
                        event.metadata["cutoff"] = int(
                            (event.resolve_date - timedelta(seconds=86400)).timestamp()
                        )
                        self.save_event(event, commit=False, cursor=c)
                c.execute(
                    """
                    update events set market_type = 'ifgames', unique_event_id = 'ifgames-' || substring(unique_event_id, INSTR(unique_event_id, '-') +  1)
                    where unique_event_id in ({subs})
                    """.format(
                        subs=",".join("?" * len(unique_event_ids))
                    ),
                    unique_event_ids,
                )
                print("Migrated pending/non-processed events: ", len(unique_event_ids))
                print("Migration is in progress..")

                count_result = c.execute(
                    """
                    select count(*) from predictions
                    where unique_event_id in ({subs})
                    """.format(
                        subs=",".join("?" * len(unique_event_ids))
                    ),
                    unique_event_ids,
                )
                print("Total predictions to migrate: ", count_result.fetchall()[0][0])
                print("Migrating predictions.. please wait..")
                now = time.perf_counter()
                c.execute(
                    """
                    update predictions set unique_event_id = 'ifgames-' || substring(unique_event_id, INSTR(unique_event_id, '-') +  1)
                    where unique_event_id in ({subs})
                    """.format(
                        subs=",".join("?" * len(unique_event_ids))
                    ),
                    unique_event_ids,
                )
                after_now = time.perf_counter()
                print("Predictions migrated. Took: ", int(after_now - now), " seconds")
                print("Migration finished ✅")

                # c.execute(
                #     """
                #     delete from predictions where rowid in (
                #         select p.rowid from predictions p inner join events e
                #         on e.unique_event_id = p.unique_event_id
                #         where e.status = '3' and e.registered_date <  date('now', '-2 months')
                #     ) and exported = '1'
                #     """
                # )
                # c.execute(
                #     """
                #     delete from events
                #     where status = '3' and registered_date <  date('now', '-2 months')
                #     and exported = '1'
                #     """
                # )
                # bt.logging.info('Cleaned old records..')
                break
            except Exception as e:
                if "locked" in str(e):
                    bt.logging.warning(f"Database locked, retry {tried + 1}..")
                    time.sleep(1 + (2 * tried))
                elif "malformed" in str(e):
                    bt.logging.warning(f"Database is malformed or locked, retry {tried + 1}..")
                    time.sleep(1 + (2 * tried))
                else:
                    bt.logging.error(f"Error during migrations: {repr(e)}")
                    bt.logging.error(traceback.format_exc())
                    self.error(
                        "We cannot proceed because of the migration issues, please reach out to Infinite Games subnet developers ❌"
                    )
                    exit(1)
                    break

            tried += 1

        conn.commit()
        conn.close()

    def sync_miners(self, axons: List[Tuple[int, AxonInfo]], blocktime: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        tries = 4
        tried = 0
        bt.logging.info("Sync miner nodes..")
        miners_len = miner_count_in_db(self.db_path)
        if miners_len == 0:
            bt.logging.info("No miners, registering all..")
        while tried < tries:
            try:
                cursor.executemany(
                    """
                    INSERT into miners ( miner_hotkey, miner_uid, node_ip, registered_date,last_updated,blocktime,blocklisted)
                    Values (?, ?, ?, ?, datetime('now', 'utc'), ?, ?)
                    ON CONFLICT(miner_hotkey, miner_uid)
                    DO UPDATE set node_ip = ?, last_updated = datetime('now', 'utc'), blocktime = ?""",
                    (
                        (
                            axon.hotkey,
                            uid,
                            axon.ip,
                            (
                                datetime.now()
                                if miners_len > 0
                                else datetime(year=2024, month=1, day=1)
                            ),
                            blocktime,
                            False,
                            axon.ip,
                            blocktime,
                        )
                        for uid, axon in axons
                    ),
                )
                conn.execute("COMMIT")
                bt.logging.info("Miner info synced.")
                break
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    bt.logging.warning(f"Database locked, retry {tried + 1}..")
                    time.sleep(1 + (2 * tried))
                    # tried += 1
                else:
                    bt.logging.error(f"Error syncing miner predictions {blocktime}: {repr(e)}")
                    bt.logging.error(traceback.format_exc())
                    break
            except Exception as e:
                bt.logging.error(f"Error syncing miners: {repr(e)}")
                bt.logging.error(traceback.format_exc())
                break
            tried += 1
        conn.close()

    def save_event(self, pe: ProviderEvent, processed=False, commit=True, cursor=None) -> bool:
        """Returns true if new event"""
        if not cursor:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
        else:
            conn = cursor.connection
            c = cursor

        result = []
        tries = 4
        tried = 0
        while tried < tries:
            # bt.logging.info(f'Now time: {datetime.now(tz=timezone.utc)}, {repr(pe)} ')
            try:
                result = c.execute(
                    """
                    INSERT into events ( unique_event_id, event_id, market_type, registered_date, description,starts, resolve_date, outcome,local_updated_at,status, metadata)
                    Values (?, ?, ?, ?, ?, ?, ?, ?, ?, ? , ?)
                    ON CONFLICT(unique_event_id)
                    DO UPDATE set outcome = ?, status = ?, local_updated_at = ?, processed = ?, metadata = ?, description = ?
                    RETURNING unique_event_id, registered_date, local_updated_at
                    """,
                    (
                        self.event_key(pe.market_type, event_id=pe.event_id),
                        pe.event_id,
                        pe.market_type,
                        pe.registered_date,
                        pe.description,
                        pe.starts,
                        pe.resolve_date,
                        pe.answer,
                        pe.registered_date,
                        pe.status,
                        json.dumps(pe.metadata),
                        pe.answer,
                        pe.status,
                        datetime.now(tz=timezone.utc),
                        processed,
                        json.dumps(pe.metadata),
                        pe.description,
                    ),
                )
                result = result.fetchall()
                # bt.logging.debug(result)
                if commit:
                    conn.execute("COMMIT")
                break
            except Exception as e:
                if "locked" in str(e):
                    bt.logging.warning(f"Database locked, retry {tried + 1}..")
                    time.sleep(1 + (2 * tried))

                else:
                    bt.logging.error(f"Error saving event {repr(pe)}: {repr(e)}")
                    bt.logging.error(traceback.format_exc())
                    break
            tried += 1
        if not cursor:
            conn.close()
        if result and result[0]:
            # bt.logging.debug(result)
            return result[0][1] == result[0][2]
        return False

    def mark_event_as_exported(self, pe: ProviderEvent) -> bool:
        """Returns true if exported successfully"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        tries = 4
        tried = 0
        while tried < tries:
            # bt.logging.info(f'Now time: {datetime.now(tz=timezone.utc)}, {repr(pe)} ')
            try:
                cursor.execute(
                    """
                    UPDATE events set exported = true
                    where unique_event_id = ?
                    """,
                    (self.event_key(pe.market_type, event_id=pe.event_id),),
                )
                # bt.logging.debug(result)
                conn.execute("COMMIT")
                conn.close()
                return True
            except Exception as e:
                if "locked" in str(e):
                    bt.logging.warning(f"Database locked, retry {tried + 1}..")
                    time.sleep(1 + (2 * tried))

                else:
                    bt.logging.error(f"Error marking event as exported {repr(pe)}: {repr(e)}")
                    bt.logging.error(traceback.format_exc())
                    break
            tried += 1

        conn.close()
        return False

    def mark_submissions_as_exported(self) -> bool:
        """Returns true if submitted successfully"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        tries = 4
        tried = 0
        while tried < tries:
            try:
                cursor.execute(
                    """
                    UPDATE predictions set exported = true
                    """,
                )
                # bt.logging.debug(result)
                conn.execute("COMMIT")
                break
            except Exception as e:
                if "locked" in str(e):
                    bt.logging.warning(f"Database locked, retry {tried + 1}..")
                    time.sleep(1 + (2 * tried))

                else:
                    bt.logging.error(f"Error marking submissions as exported: {repr(e)}")
                    bt.logging.error(traceback.format_exc())
                    break
            tried += 1

        conn.close()
        return True

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
                (self.event_key(pe.market_type, event_id=pe.event_id),),
            )
            result: List[sqlite3.Row] = c.fetchall()
        except Exception as e:
            bt.logging.error(f"Error fetching event predictions {repr(pe)}: {repr(e)}")
            bt.logging.error(traceback.format_exc())
        conn.close()
        output = defaultdict(dict)
        for row in result:
            interval_prediction = dict(row)
            if (
                int(interval_prediction["interval_start_minutes"])
                not in output[int(interval_prediction["minerUid"])]
            ):
                output[int(interval_prediction["minerUid"])][
                    int(interval_prediction["interval_start_minutes"])
                ] = interval_prediction
        return output

    def get_non_exported_event_predictions(self, pe: ProviderEvent):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        result = []
        try:
            c = cursor.execute(
                """
                select unique_event_id, minerHotkey, minerUid, predictedOutcome,interval_start_minutes,interval_agg_prediction,interval_count,submitted,blocktime
                from predictions
                where unique_event_id = ? and exported = false
                """,
                (self.event_key(pe.market_type, event_id=pe.event_id),),
            )
            result: List[sqlite3.Row] = c.fetchall()
        except Exception as e:
            bt.logging.error(f"Error fetching non-exported event predictions {repr(pe)}: {repr(e)}")
            bt.logging.error(traceback.format_exc())
        conn.close()
        output = defaultdict(dict)
        for row in result:
            interval_prediction = dict(row)
            if (
                int(interval_prediction["interval_start_minutes"])
                not in output[int(interval_prediction["minerUid"])]
            ):
                output[int(interval_prediction["minerUid"])][
                    int(interval_prediction["interval_start_minutes"])
                ] = interval_prediction
        return output

    def get_all_non_exported_event_predictions(self, interval_minutes):
        conn = sqlite3.connect(self.db_path)
        # conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        result = []
        try:
            c = cursor.execute(
                """
                select e.metadata, e.unique_event_id, p.minerHotkey, p.minerUid, p.predictedOutcome, p.interval_start_minutes, p.interval_agg_prediction,p.interval_count,p.submitted,p.blocktime
                from predictions p join events e on p.unique_event_id = e.unique_event_id
                where
                p.exported = false and interval_start_minutes = ?
                """,
                (interval_minutes,),
            )
            result: list = c.fetchall()
        except Exception as e:
            bt.logging.error(f"Error fetching all non-exported event predictions: {repr(e)}")
            bt.logging.error(traceback.format_exc())
            return []
        conn.close()
        output = result
        return output

    def miner_predict_payload_process(self, payload: pd.DataFrame):
        details_stats = json.dumps(payload["details"].value_counts().to_dict())
        bt.logging.info(f"New batch of miner predictions to be inserted: {details_stats}")

        # keep only details = valid, drop the rest
        payload = payload[payload["details"] == "valid"].copy()

        # Extract 'market_type' from 'provider_event'
        payload["meta_market_type"] = [
            pe.metadata.get("market_type", pe.market_type) for pe in payload["provider_event"]
        ]

        # if metadata market is 'azuro' overwrite interval_start_minutes with 0
        payload.loc[payload["meta_market_type"] == "azuro", "interval_start_minutes"] = 0

        payload["market_type"] = [pe.market_type for pe in payload["provider_event"]]
        payload["event_id"] = [pe.event_id for pe in payload["provider_event"]]
        payload["unique_event_id"] = [
            self.event_key(market_type, event_id)
            for market_type, event_id in zip(payload["market_type"], payload["event_id"])
        ]

        # drop unnecessary columns
        payload.drop(
            columns=["provider_event", "details", "event_id", "meta_market_type"], inplace=True
        )

        payload = payload.astype(
            {
                "unique_event_id": "str",
                "minerUid": "str",  # minerUid is string in the database
                "interval_start_minutes": "Int64",  # use nullable integer, not 'int'
                "answer": "float",
                "blocktime": "Int64",
            }
        )

        payload = payload.where(pd.notnull(payload), None)
        return payload

    @backoff.on_exception(
        backoff.expo,
        sqlite3.OperationalError,
        max_time=60,
        on_giveup=lambda details: bt.logging.error(
            f"Giving up batch DB insert for miners after {details['tries']} attempts"
        ),
    )
    def miner_batch_update_predictions(self, payload: pd.DataFrame):
        payload_processed = self.miner_predict_payload_process(payload)

        if payload_processed.empty:
            bt.logging.warning("No valid predictions to insert.")
            return

        update_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        payload_values = [
            (
                row["unique_event_id"],
                None,
                row["minerUid"],
                None,
                row["interval_start_minutes"],
                row["answer"],
                1,
                update_time,
                row["blocktime"],
                row["answer"],
            )
            for _, row in payload_processed.iterrows()
        ]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            sql_statement = """
                INSERT INTO predictions (
                    unique_event_id,
                    minerHotkey,
                    minerUid,
                    predictedOutcome,
                    interval_start_minutes,
                    interval_agg_prediction,
                    interval_count,
                    submitted,
                    blocktime
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(unique_event_id,  interval_start_minutes, minerUid)
                DO UPDATE SET
                    interval_agg_prediction = (interval_agg_prediction * interval_count + ?) / (interval_count + 1),
                    interval_count = interval_count + 1
            """

            bt.logging.debug(f"Inserting a batch of {len(payload_values)} predictions")
            cursor.executemany(sql_statement, payload_values)
            conn.commit()
        except Exception as e:
            bt.logging.error(f"Error updating miner batch prediction: {repr(e)}", exc_info=True)
            # for backoff retry only of the DB operation
            raise sqlite3.OperationalError("Error updating miner batch prediction")
        finally:
            conn.close()
