import asyncio
from typing import Optional

import aiohttp


class PolymarketProviderIntegration:
    def __init__(
        self,
    ) -> None:
        self.base_url = "https://clob.polymarket.com"
        self.lock = asyncio.Lock()
        self.loop = None

    async def _ainit(self) -> "PolymarketProviderIntegration":
        self.loop = asyncio.get_event_loop()
        return self

    async def _lock(self, seconds, error_resp):
        if not self.lock.locked():
            self.log(f"Hit limit for {error_resp.url} polymarket waiting for {seconds} seconds..")
            return await self.lock.acquire()

    async def _wait_for_retry(self, retry_seconds, resp):
        self.loop.create_task(self._lock(retry_seconds, resp))
        await asyncio.sleep(int(retry_seconds) + 1)
        self.log("Continue requests..")
        try:
            self.lock.release()
        except RuntimeError:
            pass

    async def _handle_429(self, request_resp):
        retry_timeout = request_resp.headers.get("Retry-After")
        if retry_timeout:
            if not self.lock.locked():
                await self._wait_for_retry(retry_timeout, request_resp)
            await asyncio.sleep(int(retry_timeout) + 1)
        else:
            self.log("got 429 for {request_resp} but no retry after header present.")
        return

    async def get_event_by_id(self, event_id) -> Optional[dict]:
        return await self._request(self.base_url + "/markets/{}".format(event_id))

    async def _request(self, url, max_retries=3, expo_backoff=2):
        while self.lock.locked():
            await asyncio.sleep(1)
        retried = 0
        error_response = ""
        while retried < max_retries:
            # to keep up/better sync with lock of other requests
            await asyncio.sleep(0.1)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            error_response = resp.content
                            if retried >= max_retries:
                                return
                            if resp.status == 429:
                                await self._handle_429(resp)

                        else:
                            payload = await resp.json()
                            return payload
            except Exception as e:
                error_response = str(e)
                if retried >= max_retries:
                    self.error(e)

            finally:
                retried += 1
                await asyncio.sleep(1 + retried * expo_backoff)

        self.error(f"Unable to get response {url} {error_response}")
