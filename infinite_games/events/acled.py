
from typing import AsyncIterator, Optional
import aiohttp
import asyncio
from datetime import datetime, timezone
import bittensor

from infinite_games.events.base import (
    EventStatus, ProviderEvent, ProviderIntegration
)

class AcledProviderIntegration(ProviderIntegration):
    def __init__(self, max_pending_events=None) -> None:
        super().__init__(max_pending_events=max_pending_events)
        self.base_url = 'https://ifgames.win'
        self.session = aiohttp.ClientSession()
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_event_loop()

    async def _ainit(self) -> 'AcledProviderIntegration':
        return self

    def provider_name(self):
        return 'acled'

    def latest_submit_date(self, pe: ProviderEvent):
        cutoff = pe.metadata.get('cutoff')
        cutoff = datetime.fromtimestamp(cutoff, tz=timezone.utc)
        return cutoff

    def available_for_submission(self, pe: ProviderEvent):
        max_date_for_submission = self.latest_submit_date(pe)
        # self.log(f'Can submit? {pe} resolve date: {pe.resolve_date} , condition: {datetime.now().date()} < {one_day_before_resolve.date()} {datetime.now().date() < one_day_before_resolve.date()}')
        return datetime.now(timezone.utc) < max_date_for_submission and pe.status != EventStatus.SETTLED

    def convert_status(self, event):
        if event.get('answer') is not None:
            return EventStatus.SETTLED
        else:
            return EventStatus.PENDING

    def _get_answer(self, event):
        return event.get('answer')

    def construct_provider_event(self, event_id, event):
        end_date_ts = event.get('end_date')
        start_date = event.get('start_date')
        # end_date = datetime.fromtimestamp(end_date_ts, tz=timezone.utc)
        start_date = datetime.fromtimestamp(start_date, tz=timezone.utc)
        cutoff = event.get('cutoff')
        cutoff = datetime.fromtimestamp(cutoff, tz=timezone.utc)
        
        return ProviderEvent(
            event_id,
            datetime.now(timezone.utc),
            self.provider_name(),
            event.get('title') + event.get('description'),
            start_date,
            None,  # end_date,
            self._get_answer(event),
            datetime.now(timezone.utc),
            self.convert_status(event),
            {},
            {
                'cutoff': event.get('cutoff'),
                'end_date': end_date_ts
            }
        )

    async def _lock(self, seconds, error_resp):
        if not self.lock.locked():
            self.log(f'Hit limit for {error_resp.url} polymarket waiting for {seconds} seconds..')
            return await self.lock.acquire()

    async def _wait_for_retry(self, retry_seconds, resp):
        self.loop.create_task(self._lock(retry_seconds, resp))
        await asyncio.sleep(int(retry_seconds) + 1)
        self.log('Continue requests..')
        try:
            self.lock.release()
        except RuntimeError:
            pass

    async def _handle_429(self, request_resp):
        retry_timeout = request_resp.headers.get('Retry-After')
        if retry_timeout:
            if not self.lock.locked():
                await self._wait_for_retry(retry_timeout, request_resp)
            await asyncio.sleep(int(retry_timeout) + 1)
        else:
            self.log('got 429 for {request_resp} but no retry after header present.')
        return

    async def get_event_by_id(self, event_id) -> Optional[dict]:
        return await self._request(self.base_url + '/api/events/{}'.format(event_id))

    async def get_single_event(self, event_id) -> Optional[ProviderEvent]:
        payload: Optional[dict] = await self.get_event_by_id(event_id)
        if not payload:
            return None
        pe: Optional[ProviderEvent] = self.construct_provider_event(event_id, payload)
        bittensor.logging.info(f'Retrieved event: {pe} {pe.status} {pe.starts}')
        return pe

    async def _request(self, url, max_retries=3, expo_backoff=2):
        while self.lock.locked():
            await asyncio.sleep(1)
        retried = 0
        error_response = ''
        while retried < max_retries:
            # to keep up/better sync with lock of other requests
            await asyncio.sleep(0.1)
            try:
                async with self.session.get(url) as resp:
                    if resp.status != 200:
                        error_response = resp.content
                        if retried >= max_retries:
                            return
                        if resp.status == 429:
                            await self._handle_429(resp)
                        # self.log(f'Retry {url}.. {retried + 1}')
                    else:

                        payload = await resp.json()
                        return payload
            except Exception as e:
                error_response = str(e)
                if retried >= max_retries:
                    self.error(e)
                    # return
            finally:
                retried += 1
                await asyncio.sleep(1 + retried * expo_backoff)

        self.error(f'Unable to get response {url} {error_response}')

    async def sync_events(self, start_from: int = None) -> AsyncIterator[ProviderEvent]:
        self.log(f"syncing events {start_from=} ")
        resp = await self._request(self.base_url + '/api/events')

        if resp:
            for event in resp["items"]:
                pe = self.construct_provider_event(event['event_id'], event)
                if not pe:
                    continue
                if not self.available_for_submission(pe):
                    continue
                yield pe
        else:
                return
