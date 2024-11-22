
import os
import traceback
from typing import AsyncIterator, Optional
import aiohttp
import backoff
import bittensor as bt
import asyncio
from datetime import datetime, timedelta, timezone
import requests

from infinite_games.events.base import EventStatus, ProviderEvent, ProviderIntegration


class PolymarketProviderIntegration(ProviderIntegration):
    def __init__(self, max_pending_events=None) -> None:
        super().__init__(max_pending_events=max_pending_events)
        self.base_url = 'https://clob.polymarket.com'
        self.session = aiohttp.ClientSession()
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_event_loop()

    async def _ainit(self) -> 'PolymarketProviderIntegration':
        return self

    def provider_name(self):
        return 'polymarket'

    def latest_submit_date(self, pe: ProviderEvent):
        return pe.resolve_date - timedelta(seconds=86400)

    def available_for_submission(self, pe: ProviderEvent):
        max_date_for_submission = self.latest_submit_date(pe)
        # self.log(f'Can submit? {pe} resolve date: {pe.resolve_date} , condition: {datetime.now().date()} < {one_day_before_resolve.date()} {datetime.now().date() < one_day_before_resolve.date()}')
        return datetime.now(timezone.utc) < max_date_for_submission and pe.status != EventStatus.DISCARDED

    def convert_status(self, closed_bool):
        return {
            True: EventStatus.SETTLED,
            False: EventStatus.PENDING,
        }.get(closed_bool, EventStatus.PENDING)

    def _get_answer(self, market):
        """
        Example of answers

        'tokens': [{'outcome': 'Yes',
             'price': 0.195,
             'token_id': '49890085726756071917943790579691800691766085550672445428078676220814757758014',
             'winner': False},
            {'outcome': 'No',
             'price': 0.805,
             'token_id': '59333345431942443015299430667133768751255782539994157405171948134198386135362',
             'winner': False}]}
        """

        toks = market["tokens"]
        if toks[0]["winner"]:
            return 1
        elif toks[1]["winner"]:
            return 0
        else:
            return None

    def construct_provider_event(self, event_id, market):
        payload = market
        if not payload.get('end_date_iso'):
            self.log(f'Skip event without end date: {market["market_slug"]}!')
            # pprint(payload)
            return
        end_date_iso = payload.get('end_date_iso')
        if end_date_iso:
            end_date_iso = end_date_iso.replace('Z', '+00:00')
        resolve_date = datetime.fromisoformat(end_date_iso).replace(tzinfo=timezone.utc)
        start_date = None

        if payload['game_start_time']:
            start_date = datetime.fromisoformat(payload['game_start_time'].replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
        # from pprint import pprint
        # pprint(market)
        # if 'will-scottie-scheffler-win-the-us-open' in market.get('market_slug'):
        #     self.log(f'(DEBUG) {market.get('condition_id')} FORCE TEST CLOSE')
        #     payload['closed'] = True
        #     payload["tokens"][0]["winner"] = True

        return ProviderEvent(
            event_id,
            datetime.now(timezone.utc),
            self.provider_name(),
            payload.get('question') + '.' + payload.get('description'),
            start_date,
            resolve_date,
            self._get_answer(payload),
            datetime.now(timezone.utc),
            self.convert_status(payload.get('closed')),
            {},
            {
                'description': payload.get('description'),
                'category': payload.get('category'),
                'active': payload.get('active'),
                'slug': payload.get('market_slug'),
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
        return await self._request(self.base_url + '/markets/{}'.format(event_id))

    async def get_single_event(self, event_id) -> Optional[ProviderEvent]:
        payload: Optional[dict] = await self.get_event_by_id(event_id)
        if not payload:
            return None
        pe: Optional[ProviderEvent] = self.construct_provider_event(event_id, payload)
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
                # self.log(f'Retry {url}.. {retried + 1}')
            finally:
                retried += 1
                await asyncio.sleep(1 + retried * expo_backoff)

        self.error(f'Unable to get response {url} {error_response}')

    async def sync_events(self, start_from: int = None) -> AsyncIterator[ProviderEvent]:
        self.log(f"syncing events {start_from=} ")
        if not start_from:
            start_from = int(datetime.now().timestamp())
        first = True
        cursor = None
        max_events = 20000
        count = 0

        while cursor != "LTE=":
            resp = None
            if first:
                try:
                    resp = await self._request("https://clob.polymarket.com/sampling-markets")
                    nxt = resp
                    first = False
                except Exception as e:
                    self.error(str(e))
            else:
                try:
                    resp = await self._request("https://clob.polymarket.com/sampling-markets?next_cursor={}".format(cursor))
                    nxt = resp
                except Exception as e:
                    self.error(str(e))

            if resp:

                cursor = nxt["next_cursor"]
                for market in nxt["data"]:
                    if count > max_events:
                        return
                    count += 1
                    if not market.get('condition_id'):
                        self.error('No market id provided for event {market}')
                        continue
                    try:
                        pe = self.construct_provider_event(market.get('condition_id'), market)
                        if not pe:
                            continue
                        if not self.available_for_submission(pe):
                            # self.log(f'Settle is {pe.resolve_date.date()} , ignore event {pe}')
                            continue
                        # for Polymarket we only fetch next 2-3 days events
                        # self.log(f'{pe.resolve_date.date()} limited to? {datetime.now().date() + timedelta(days=3)} {datetime.now().date() + timedelta(days=3) > pe.resolve_date.date()}')
                        if datetime.now(timezone.utc).date() + timedelta(days=14) < pe.resolve_date.date():
                            continue
                        yield pe
                    except Exception as e:

                        self.error(f"Error parse market {market.get('market_slug')} {e} {market}")
                        self.error(traceback.format_exc())
            else:
                return
            await asyncio.sleep(15)
