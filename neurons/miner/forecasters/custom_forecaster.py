import os
import time
import openai
from neurons.miner.forecasters.base import BaseForecaster

# Simple in-memory cache: { event_uid: (timestamp, probability) }
_CACHE = {}
_CACHE_TTL = 4 * 60 * 60  # 4 hours

class CustomForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def _run(self, event):
        uid = event.uid
        now = time.time()

        # 1) Return cached value if still fresh
        if uid in _CACHE:
            ts, prob = _CACHE[uid]
            if now - ts < _CACHE_TTL:
                return prob

        # 2) Build and send prompt
        prompt = self._build_prompt(event)
        resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        # 3) Parse the LLM’s numeric answer
        prob = float(resp.choices[0].message.content.strip())

        # 4) Cache and return
        _CACHE[uid] = (now, prob)
        return prob

    def _build_prompt(self, event):
        return (
            f"You are a super-forecasting expert.\n\n"
            f"Event: {event.question}\n"
            f"Current probability: {event.current_probability:.2f}\n\n"
            "Step by step, reason through factors that could move this, "
            "then give your final forecast as a number between 0 and 1:"
        )
