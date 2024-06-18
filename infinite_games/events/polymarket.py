
from typing import AsyncIterator, Optional
import aiohttp
import backoff

import asyncio
from datetime import datetime, timedelta
import json
import requests
from pprint import pprint

from infinite_games.events.base import EventStatus, ProviderEvent, ProviderIntegration


class PolymarketProviderIntegration(ProviderIntegration):
    def __init__(self, max_pending_events=None) -> None:
        super().__init__(max_pending_events=max_pending_events)
        self.base_url = 'https://clob.polymarket.com'
        self.session = aiohttp.ClientSession()

    async def _ainit(self) -> 'PolymarketProviderIntegration':
        return self

    def provider_name(self):
        return 'polymarket'

    def available_for_submission(self, pe: ProviderEvent):
        return datetime.now().date() < pe.resolve_date.date()

    def convert_status(self, closed_bool):
        return {
            True: EventStatus.SETTLED,
            False: EventStatus.PENDING,
        }.get(closed_bool, EventStatus.PENDING)

    def _get_answer(self, market):
        toks = market["tokens"]
        if toks[0]["winner"]:
            return 1
        elif toks[1]["winner"]:
            return 2
        else:
            return None

    def construct_provider_event(self, event_id, market):
        payload = market
        if not payload.get('end_date_iso'):
            self.error(f'No end date for event: {market["market_slug"]}!')
            # pprint(payload)
            return
        end_date_iso = payload.get('end_date_iso')
        if end_date_iso:
            end_date_iso.replace('Z', '+00:00')
        resolve_date = datetime.fromisoformat(end_date_iso).replace(tzinfo=None)
        start_date = None

        if payload['game_start_time']:
            start_date = datetime.fromisoformat(payload['game_start_time'].replace('Z', '+00:00')).replace(tzinfo=None)

        # if 'will-scottie-scheffler-win-the-us-open' in market.get('market_slug'):
        #     self.log(f'(DEBUG) {market.get('condition_id')} FORCE TEST CLOSE')
        #     payload['closed'] = True
        #     payload["tokens"][0]["winner"] = True

        return ProviderEvent(
            event_id,
            self.provider_name(),
            payload.get('question'),
            start_date,
            resolve_date,
            self._get_answer(payload),
            datetime.now(),
            self.convert_status(payload.get('closed')),
            {},
            {
                'description': payload.get('description'),
                'category': payload.get('category'),
                'active': payload.get('active'),
                'slug': payload.get('market_slug'),
            }
        )

    @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def get_single_event(self, event_id) -> ProviderEvent:
        market_url = self.base_url + '/markets/{}'.format(event_id)
        async with self.session.get(market_url) as resp:
            payload = await resp.json()
            if not payload or not payload.get('condition_id'):
                self.error(f'no condition id for {event_id=} response: {payload}')
                return
        pe: Optional[ProviderEvent] = self.construct_provider_event(event_id, payload)
        return pe

    @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def sync_events(self, start_from: int = None) -> AsyncIterator[ProviderEvent]:
        self.log(f"syncing events {start_from=} ")
        if not start_from:
            start_from = int(datetime.now().timestamp())

        first = True
        cursor = None
        max_events = 5000
        count = 0

        while cursor != "LTE=":
            if first:
                resp = requests.get("https://clob.polymarket.com/sampling-markets")
                nxt = resp.json()
                first = False
            else:
                resp = requests.get("https://clob.polymarket.com/sampling-markets?next_cursor={}".format(cursor))
                nxt = resp.json()

            if resp.status_code == 200:

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
                        if datetime.now().date() + timedelta(days=3) < pe.resolve_date.date():
                            continue
                        yield pe
                    except Exception as e:

                        self.error(f"Error parse market {market.get('market_slug')} {e} {market}")
            await asyncio.sleep(2)
