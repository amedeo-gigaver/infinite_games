from datetime import datetime

import backoff
from infinite_games.events.base import EventStatus, ProviderEvent, ProviderIntegration
from tests.utils import after


class MockAzuroProviderIntegration(ProviderIntegration):
    def __init__(self, max_pending_events=None) -> None:
        super().__init__(max_pending_events=max_pending_events)
        self.session = None
        # self.ws_connection = None
        # asyncio.create_task()

    async def _ainit(self) -> 'MockAzuroProviderIntegration':
        pass

    def provider_name(self):
        return 'azuro'

    async def _connect(self):
        self.session = await self.client.connect_async(reconnecting=True)

    async def _close(self):
        self.session = await self.client.close_async()

    def available_for_submission(self, pe: ProviderEvent) -> bool:
        # self.log(f'Can submit? {pe} {pe.starts} > {datetime.now()} {pe.starts > datetime.now()}')
        return pe.starts > datetime.now() and pe.status != EventStatus.DISCARDED

    def convert_status(self, azuro_status):
        return {
            'Created': EventStatus.PENDING,
            'Resolved': EventStatus.SETTLED,
            'Canceled': EventStatus.DISCARDED,
            'Paused': EventStatus.PENDING
        }.get(azuro_status, EventStatus.PENDING)


    def _get_answer(self, status):
        if status == 'Won':
            return 1
        elif status == 'Lost':
            return 0
        else:
            return None

    async def get_event_by_id(self, event_id) -> dict:
        
        return None

    async def get_single_event(self, event_id) -> ProviderEvent:
        start_date = datetime.now() + after(hours=4)
        event_status = 'Created'
        effective_status = self.convert_status(event_status)
        answer = None

        # if event_id == '0x7f3f3f19c4e4015fd9db2f22e653c766154091ef_100100000000000015927405030000000000000357953524_142':
        #     effective_status = EventStatus.SETTLED
        #     answer = 1
        pe = ProviderEvent(
            event_id,
            self.provider_name(),
            'Fake title' + ' ,' + ' Comment of match',
            start_date,
            None,
            answer,
            datetime.now(),
            effective_status,
            None,
            {
                'conditionId': 'cond id',
                'slug': 'slug',
                'league': 'league'
            }
        )
        return pe

    # @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def sync_events(self, start_from: int = None):
        return None