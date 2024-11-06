
from typing import AsyncIterator

import backoff
import bittensor as bt

from datetime import datetime, timedelta, timezone
import json
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import websockets

from infinite_games.azurodictionaries.outcomes import OUTCOMES

from infinite_games.events.base import (
    EventAggregator, EventStatus, ProviderEvent, ProviderIntegration
)


class AzuroProviderIntegration(ProviderIntegration):
    def __init__(self, max_pending_events=None) -> None:
        super().__init__(max_pending_events=max_pending_events)
        self.transport = AIOHTTPTransport(
            url="https://thegraph.azuro.org/subgraphs/name/azuro-protocol/azuro-api-gnosis-v3"
        )
        self.client = Client(transport=self.transport, fetch_schema_from_transport=True)
        self.session = None
        self._listen_stream_url = 'wss://streams.azuro.org/v1/streams/conditions'
        # self.ws_connection = None
        # asyncio.create_task()

    async def _ainit(self) -> 'AzuroProviderIntegration':
        await self._connect()
        return self

    def provider_name(self):
        return 'azuro'

    async def _connect(self):
        self.session = await self.client.connect_async(reconnecting=True)

    async def _close(self):
        self.session = await self.client.close_async()

    def latest_submit_date(self, pe: ProviderEvent):
        return pe.starts

    def available_for_submission(self, pe: ProviderEvent) -> bool:
        # self.log(f'Can submit? {pe} {pe.starts} > {datetime.now()} {pe.starts > datetime.now()}')
        return self.latest_submit_date(pe) > datetime.now(timezone.utc) and pe.status != EventStatus.DISCARDED

    def convert_status(self, azuro_status):
        return {
            'Created': EventStatus.PENDING,
            'Resolved': EventStatus.SETTLED,
            'Canceled': EventStatus.DISCARDED,
            'Paused': EventStatus.PENDING
        }.get(azuro_status, EventStatus.PENDING)

    async def on_update(self, ws):
        try:
            async for message in ws:
                self.log(f'Websocket {message=}')
        except websockets.ConnectionClosed:
            self.ws_connection = None

    async def subscribe_to_condition(self, cids):
        # TODO with websocket
        if self.ws_connection:
            if cids:
                self.ws_connection.send(json.dumps(
                    {
                        "action": 'subscribe',
                        "conditionIds": cids,
                    }
                ))
            else:
                bt.logging.warning('Azuro: Empty CID passed for ws subscribe')
        else:
            bt.logging.error('Azuro: Could not subscribe to event no WS connection found!')

    # async def listen_for_updates(self):
    #     async for websocket in websockets.connect(
    #         self._listen_stream_url
    #     ):
    #         self.ws_connection = websocket
    #         asyncio.create_task(self.on_update(self, websocket))

    def _get_answer(self, status):
        if status == 'Won':
            return 1
        elif status == 'Lost':
            return 0
        else:
            return None

    @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def get_event_by_id(self, event_id) -> dict:
        query = gql(
            """
            query SingleOutcome($id: ID!) {
                outcome(id: $id) {
                    id
                    outcomeId
                    fund
                    result
                    currentOdds
                    condition {
                        id
                        conditionId
                        outcomes {
                            id
                        }
                        status
                        provider
                        game {
                            title
                            gameId
                            slug
                            startsAt
                            league {
                                name
                                slug
                                country {
                                    name
                                    slug
                                }
                            }
                            status
                            sport {
                                name
                                slug
                            }
                            participants {
                                image
                                name
                            }
                        }
                    }
                }
                }


        """
        )

        result = await self.session.execute(
            query,
            {
                "id": event_id
            }
        )
        if not result:
            bt.logging.error(f'Azuro: Could not fetch event by id  {event_id}')
            return None

        return result

    async def get_single_event(self, event_id) -> ProviderEvent:
        result = await self.get_event_by_id(event_id)
        if result is None:
            return None

        outcome = result['outcome']

        if not outcome:
            bt.logging.error(f'Azuro: Could not fetch event by id  {event_id}')
            return None
        condition = outcome['condition']
        game = condition['game']
        start_date = datetime.fromtimestamp(int(game["startsAt"]), tz=timezone.utc)
        event_status = condition.get('status')
        effective_status = self.convert_status(event_status)
        answer = self._get_answer(outcome.get('result'))

        # if event_id == '0x7f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927405030000000000000357953524_142':
        #     effective_status = EventStatus.SETTLED
        #     answer = 1
        now = datetime.now(timezone.utc)
        pe = ProviderEvent(
            event_id,
            now,
            self.provider_name(),
            game.get('title') + ' ,' + OUTCOMES[outcome['outcomeId']].get('_comment'),
            start_date,
            None,
            answer,
            now,
            effective_status,
            None,
            {
                'conditionId': condition['conditionId'],
                'slug': game.get('slug'),
                'league': game.get('league')
            }
        )
        return pe

    # @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def sync_events(self, start_from: int = None) -> AsyncIterator[ProviderEvent]:
        """For azuro we fetch first n events for tomorrow (starting at 8 UTC) only between 8 and 13 UTC"""
        MAX_DAILY_EVENTS = 20
        now = datetime.now(tz=timezone.utc)

        if not (now.hour > 7 and now.hour < 14):
            self.log(f'Skip sync h:{now.hour}')
            return
        if not start_from:
            DAILY_EVENT_START_HOURS_UTC = 8
            # if we started this day already register events for tomorrow
            now = now + timedelta(days=1)
            now = now.replace(hour=DAILY_EVENT_START_HOURS_UTC, minute=0, second=0, microsecond=0)
            self.log(f'Syncing events from {now}')
            start_from = int(now.timestamp())
        else:

            self.log(f"Syncing events {start_from=} ")
        query = gql(
            """
            query Games($where: Game_filter!, $start: Int!, $per_page: Int!) {
                games(
                    skip: $start
                    first: $per_page
                    where: $where
                    orderBy: startsAt
                    orderDirection: asc
                    subgraphError: allow
                ) {
                    gameId
                    title
                    slug
                    status
                    startsAt
                    league {
                        name
                        slug
                        country {
                            name
                            slug
                        }
                    }
                    sport {
                        name
                        slug
                    }
                    participants {
                        image
                        name
                    }
                    conditions {
                        conditionId
                        isExpressForbidden
                        status
                        outcomes {
                            id
                            currentOdds
                            outcomeId
                            result
                        }
                    }
                }
            }


        """
        )

        result = await self.session.execute(
            query,
            {
                "where": {
                    "status": "Created",
                    "hasActiveConditions": True,
                    "sport_": {
                        "name_in": ["Football", "Tennis", "Basketball"]
                    },
                    "startsAt_gt": start_from
                },
                "start": 0, "per_page": MAX_DAILY_EVENTS
            },
        )
        # self.log(f'Fetched games: {len(result["games"])}')
        # from pprint import pprint
        # pprint(result['games'])
        max_outcome_per_game = 1
        for game in result["games"]:
            game_events = 0
            start_date = datetime.fromtimestamp(int(game["startsAt"]), tz=timezone.utc)
            if not game.get('startsAt'):
                bt.logging.warning(f"Azuro game {game.get('slug')} doesnt have start time, skipping..")
                continue
            for condition in game['conditions']:
                if game_events >= max_outcome_per_game:
                    break
                event_status = condition.get('status')
                # if not event_status != 'Canceled':
                #     bt.logging.debug(f"Azuro condition for game {game.get('slug')} condition id {condition.get('conditionId')} is {condition.get('status')}, skipping..")
                #     continue
                if event_status == 'Canceled':
                    continue
                for outcome in condition['outcomes']:
                    if game_events >= max_outcome_per_game:
                        break
                    game_events += 1
                    if outcome.get('id') is None:
                        bt.logging.error(f"{game.get('slug')} cid: {condition.get('conditionId')} outcome {outcome.get('outcomeId')} does not have id, skip..")
                        continue
                    if outcome.get('result') is not None:
                        bt.logging.debug(f"{game.get('slug')} cid: {condition.get('conditionId')} outcome {outcome.get('outcomeId')} resolved, skipping..")
                        continue
                    yield ProviderEvent(
                        outcome.get('id'),
                        datetime.now(timezone.utc),
                        self.provider_name(),
                        game.get('title') + ' ,' + OUTCOMES[outcome['outcomeId']].get('_comment'),
                        start_date,
                        resolve_date=None,
                        answer=None,
                        local_updated_at=datetime.now(timezone.utc),
                        status=self.convert_status(event_status),
                        miner_predictions={},
                        metadata={
                            'conditionId': condition['conditionId'],
                            'slug': game.get('slug'),
                            'league': game.get('league')
                            },
                    )
                # TODO test websocket
                # self.subscribe_to_condition(condition.get('conditionId'))
