import backoff
import bittensor as bt
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport


class AzuroProviderIntegration:
    def __init__(self) -> None:
        self.transport = AIOHTTPTransport(
            url="https://thegraph.azuro.org/subgraphs/name/azuro-protocol/azuro-api-gnosis-v3"
        )
        self.client = Client(transport=self.transport, fetch_schema_from_transport=True)
        self.session = None

    async def _ainit(self) -> "AzuroProviderIntegration":
        await self._connect()
        return self

    async def _connect(self):
        self.session = await self.client.connect_async(reconnecting=True)

    async def _close(self):
        self.session = await self.client.close_async()

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

        result = await self.session.execute(query, {"id": event_id})

        if not result:
            bt.logging.error(f"Azuro: Could not fetch event by id  {event_id}")
            return None

        return result
