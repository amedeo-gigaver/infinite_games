import asyncio
import sys
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import aiohttp
import bittensor

from infinite_games.events.base import (
    EventRemovedException,
    EventStatus,
    ProviderEvent,
    ProviderIntegration,
)


class IFGamesProviderIntegration(ProviderIntegration):
    def __init__(self, max_pending_events=None) -> None:
        super().__init__(max_pending_events=max_pending_events)

        self.lock = asyncio.Lock()
        self.loop = None
        self.is_test = "subtensor.networktest" in "".join(sys.argv)
        self.base_url = "https://stage.ifgames.win" if self.is_test else "https://ifgames.win"

    async def _ainit(self) -> "IFGamesProviderIntegration":
        self.session = aiohttp.ClientSession()
        self.loop = asyncio.get_event_loop()
        return self

    async def close(self):
        if self.session:
            await self.session.close()

    def provider_name(self):
        return "ifgames"

    def latest_submit_date(self, pe: ProviderEvent):
        cutoff = pe.metadata.get("cutoff") or pe.resolve_date or pe.starts
        if isinstance(cutoff, int):
            cutoff = datetime.fromtimestamp(cutoff, tz=timezone.utc)
        return cutoff

    def available_for_submission(self, pe: ProviderEvent):
        max_date_for_submission = self.latest_submit_date(pe)
        return (
            datetime.now(timezone.utc) < max_date_for_submission
            and pe.status != EventStatus.SETTLED
        )

    def convert_status(self, event):
        return EventStatus.SETTLED if event.get("answer") is not None else EventStatus.PENDING

    def _get_answer(self, event):
        return event.get("answer")

    def construct_provider_event(self, event_id, event):
        end_date_ts = event.get("end_date")
        start_date_ts = event.get("start_date")
        start_date = datetime.fromtimestamp(start_date_ts, tz=timezone.utc)
        cutoff_ts = event.get("cutoff")
        cutoff = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc)  # ?!
        return ProviderEvent(
            event_id=event_id,
            registered_date=datetime.now(timezone.utc),
            market_type=self.provider_name(),
            description=event.get("title", "") + event.get("description", ""),
            starts=start_date,
            resolve_date=None,
            answer=self._get_answer(event),
            local_updated_at=datetime.now(timezone.utc),
            status=self.convert_status(event),
            miner_predictions={},
            metadata={
                "market_type": event.get("market_type", "").lower(),
                "cutoff": event.get("cutoff"),
                "end_date": end_date_ts,
            },
        )

    async def _lock(self, seconds, error_resp):
        if not self.lock.locked():
            self.log(f"Hit rate limit for {error_resp.url}, waiting for {seconds} seconds...")
            return await self.lock.acquire()

    async def _wait_for_retry(self, retry_seconds, resp):
        self.loop.create_task(self._lock(retry_seconds, resp))
        await asyncio.sleep(int(retry_seconds) + 1)
        self.log("Continuing requests after rate limit...")
        try:
            self.lock.release()
        except RuntimeError:
            pass

    async def _handle_429(self, response):
        retry_timeout = response.headers.get("Retry-After")
        if retry_timeout:
            if not self.lock.locked():
                await self._wait_for_retry(retry_timeout, response)
            await asyncio.sleep(int(retry_timeout) + 1)
        else:
            self.error(f"Got 429 for {response.url} but no Retry-After header present.")

    async def get_event_by_id(self, event_id) -> Optional[dict]:
        return await self._request(f"{self.base_url}/api/v2/events/{event_id}")

    async def get_single_event(self, event_id) -> Optional[ProviderEvent]:
        payload = await self.get_event_by_id(event_id)
        if not payload:
            return None
        pe = self.construct_provider_event(event_id, payload)
        bittensor.logging.info(f"Retrieved event: {pe} {pe.status} {pe.starts}")
        return pe

    async def _request(self, url, max_retries=3, expo_backoff=2):
        while self.lock.locked():
            await asyncio.sleep(1)
        retried = 0
        error_response = ""
        while retried < max_retries:
            await asyncio.sleep(0.1)
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        error_response = await resp.text()
                        if resp.status == 429:
                            await self._handle_429(resp)
                        elif resp.status == 410:
                            raise EventRemovedException()
                        elif resp.status == 404 and self.is_test:
                            self.log(f"[TEST] Removing not found event from {url}")
                            raise EventRemovedException()
                        retried += 1
                        await asyncio.sleep(expo_backoff**retried)
            except EventRemovedException:
                raise
            except Exception as e:
                error_response = str(e)
                if retried >= max_retries:
                    self.error(f"Error requesting {url}: {repr(e)}")
                    return
                retried += 1
                await asyncio.sleep(expo_backoff**retried)
        self.error(f"Unable to get response from {url}: {error_response}")

    async def sync_events(self, start_from: int = None) -> AsyncIterator[ProviderEvent]:
        self.log(f"Syncing events from {start_from=}")
        if start_from is None:
            start_from = 1
        offset = 0
        while True:
            await asyncio.sleep(1)
            self.log(f"Sync events after {start_from=} {offset=}")
            resp = await self._request(
                f"{self.base_url}/api/v2/events?limit=250&from_date={start_from}&offset={offset}"
            )
            if resp and resp.get("count", 0) > 0:
                for event in resp["items"]:
                    pe = self.construct_provider_event(event["event_id"], event)
                    if pe and self.available_for_submission(pe):
                        yield pe
                offset += resp["count"]
            else:
                break
