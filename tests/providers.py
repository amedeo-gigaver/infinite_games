from datetime import datetime, timezone

from infinite_games.events.azuro import AzuroProviderIntegration
from infinite_games.events.base import EventStatus, ProviderEvent
from infinite_games.events.ifgames import IFGamesProviderIntegration
from infinite_games.events.polymarket import PolymarketProviderIntegration
from tests.utils import after


class MockAzuroProviderIntegration(AzuroProviderIntegration):
    def latest_submit_date(self, pe: ProviderEvent):
        return pe.starts

    def available_for_submission(self, pe: ProviderEvent) -> bool:
        # self.log(f'Can submit? {pe} {pe.starts} > {datetime.now()} {pe.starts > datetime.now()}')
        return pe.starts > datetime.now(timezone.utc) and pe.status != EventStatus.DISCARDED

    def convert_status(self, azuro_status):
        return {
            "Created": EventStatus.PENDING,
            "Resolved": EventStatus.SETTLED,
            "Canceled": EventStatus.DISCARDED,
            "Paused": EventStatus.PENDING,
        }.get(azuro_status, EventStatus.PENDING)

    def _get_answer(self, status):
        if status == "Won":
            return 1
        elif status == "Lost":
            return 0
        else:
            return None

    async def get_event_by_id(self, event_id) -> dict:
        return None

    async def get_single_event(self, event_id) -> ProviderEvent:
        start_date = datetime.now() + after(hours=4)
        event_status = "Created"
        effective_status = self.convert_status(event_status)
        answer = None

        # if event_id == '0x7f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927405030000000000000357953524_142':
        #     effective_status = EventStatus.SETTLED
        #     answer = 1
        pe = ProviderEvent(
            event_id,
            self.provider_name(),
            "Fake title" + " ," + " Comment of match",
            start_date,
            None,
            answer,
            datetime.now(),
            effective_status,
            None,
            {"conditionId": "cond id", "slug": "slug", "league": "league"},
        )
        return pe

    # @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def sync_events(self, start_from: int = None):
        return None


class MockPolymarketProviderIntegration(PolymarketProviderIntegration):
    def latest_submit_date(self, pe: ProviderEvent):
        return pe.resolve_date

    # @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def sync_events(self, start_from: int = None):
        return None


class MockIFGamesProviderIntegration(IFGamesProviderIntegration):
    async def _request(self, url, max_retries=1, expo_backoff=2):
        # last date return empty page
        # if '1822462600' in url:
        #     return {
        #         'count': 0,
        #         'items': []
        #     }

        if url == self.base_url + "/api/v2/events/7b787c68-d6df-4138-a10b-0de76eeec5c3":
            return {
                "event_id": "7b787c68-d6df-4138-a10b-0de76eeec5c3",
                "cutoff": 1722459000,
                "market_type": "acled",
                "title": "Will the amount of peaceful protests in Bulgaria be above 0 for the duration 2024-08-01 to 2024-08-01?",
                "description": "This event resolves to `YES` if there are at least 1 peaceful protests, for the country of Bulgaria between the following dates, from 2024-08-01  00:00:00 to 2024-08-01  23:59:59 in Europe/Sofia timezone. A peaceful protest is when demonstrators gather for a protest and do not engage in violence or other forms of rioting activity, such as property destruction, and are not met with any sort of force or intervention.",
                "start_date": 1722459600,
                "created_at": 1822462600,
                "end_date": 1722545999,
                "answer": None,
            }
        elif url == self.base_url + "/api/events/dbcba93a-fe3b-4092-b918-8231b23f2faa":
            return {
                "event_id": "dbcba93a-fe3b-4092-b918-8231b23f2faa",
                "cutoff": 1722462600,
                "market_type": "acled",
                "title": "Will the amount of peaceful protests in Belgium be above 0 for the duration 2024-08-01 to 2024-08-01?",
                "description": "This event resolves to `YES` if there are at least 1 peaceful protests, for the country of Belgium between the following dates, from 2024-08-01  00:00:00 to 2024-08-01  23:59:59 in Europe/Brussels timezone. A peaceful protest is when demonstrators gather for a protest and do not engage in violence or other forms of rioting activity, such as property destruction, and are not met with any sort of force or intervention.",
                "start_date": 1722463200,
                "created_at": 1822462600,
                "end_date": 1722549599,
                "answer": None,
            }
        elif url == self.base_url + "/api/v2/events/cbcba93a-fe3b-4092-b918-8231b23f2faa":
            return {
                "event_id": "cbcba93a-fe3b-4092-b918-8231b23f2faa",
                "cutoff": 1722462600,
                "market_type": "polymarket_prices",
                "title": "Will the repricing events..",
                "description": "repricing text",
                "start_date": 1722463200,
                "created_at": 1822462600,
                "end_date": 1722549599,
                "answer": None,
            }
        elif url == self.base_url + "/api/v2/events/dbcba93a-fe3b-4092-b918-8231b23f2faa":
            return {
                "event_id": "dbcba93a-fe3b-4092-b918-8231b23f2faa",
                "cutoff": 1722462600,
                "market_type": "uma",
                "title": "Will the price be 1.5 tomorrow",
                "description": "Will the price be 1.5 tomorrow",
                "start_date": 1722463200,
                "created_at": 1822462600,
                "end_date": 1722549599,
                "answer": None,
            }
        elif url == self.base_url + "/api/v2/events/dbcba93a-fe3b-4092-b918-8231b23f2faa":
            return {
                "event_id": "0x8f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927404810000000000000365390949_2000",
                "cutoff": 1722462600,
                "market_type": "azuro",
                "title": "Match",
                "description": "Match A B",
                "start_date": 1722463200,
                "created_at": 1822462600,
                "end_date": 1722549599,
                "answer": None,
            }
        elif self.base_url + "/api/v2/events" in url:
            return {
                "count": 4,
                "items": [
                    {
                        "event_id": "7b787c68-d6df-4138-a10b-0de76eeec5c3",
                        "cutoff": 1822462600,
                        "market_type": "acled",
                        "title": "Will the amount of peaceful protests in Bulgaria be above 0 for the duration 2024-08-01 to 2024-08-01?",
                        "description": "This event resolves to `YES` if there are at least 1 peaceful protests, for the country of Bulgaria between the following dates, from 2024-08-01  00:00:00 to 2024-08-01  23:59:59 in Europe/Sofia timezone. A peaceful protest is when demonstrators gather for a protest and do not engage in violence or other forms of rioting activity, such as property destruction, and are not met with any sort of force or intervention.",
                        "start_date": 1822462600,
                        "created_at": 1822462600,
                        "end_date": 1822462600,
                        "answer": None,
                    },
                    {
                        "event_id": "dbcba93a-fe3b-4092-b918-8231b23f2faa",
                        "cutoff": 1822462600,
                        "market_type": "acled",
                        "title": "Will the amount of peaceful protests in Belgium be above 0 for the duration 2024-08-01 to 2024-08-01?",
                        "description": "This event resolves to `YES` if there are at least 1 peaceful protests, for the country of Belgium between the following dates, from 2024-08-01  00:00:00 to 2024-08-01  23:59:59 in Europe/Brussels timezone. A peaceful protest is when demonstrators gather for a protest and do not engage in violence or other forms of rioting activity, such as property destruction, and are not met with any sort of force or intervention.",
                        "start_date": 1822462600,
                        "created_at": 1822462600,
                        "end_date": 1822462600,
                        "answer": None,
                    },
                    {
                        "event_id": "cbcba93a-fe3b-4092-b918-8231b23f2faa",
                        "cutoff": 1822462600,
                        "market_type": "polymarket_prices",
                        "title": "Will the price be 1.5 tomorrow",
                        "description": "Will the price be 1.5 tomorrow",
                        "start_date": 1822462600,
                        "created_at": 1822462600,
                        "end_date": 1822462600,
                        "answer": None,
                    },
                    {
                        "event_id": "dbcba93a-fe3b-4092-b918-8231b23f2faa",
                        "cutoff": 1822462600,
                        "market_type": "uma",
                        "title": "Will the price be 1.5 tomorrow",
                        "description": "Will the price be 1.5 tomorrow",
                        "start_date": 1822462600,
                        "created_at": 1822462600,
                        "end_date": 1822462600,
                        "answer": None,
                    },
                ],
            }
        else:
            raise ValueError(f"Not mocked or not expected url! {url}")
