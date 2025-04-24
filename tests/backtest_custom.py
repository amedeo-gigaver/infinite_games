import requests
from neurons.miner.models.event import MinerEvent
from neurons.miner.forecasters.custom_forecaster import CustomForecaster

# Fetch the last 100 resolved events (past 24 h)
URL = "https://ifgames.win/api/v2/events/resolved?resolved_since=86400&limit=100"
data = requests.get(URL).json()

errors = []
for raw in data:
    event = MinerEvent.from_dict(raw)
    pred = CustomForecaster(event=event)._run(event)
    actual = 1.0 if raw["outcome"] else 0.0
    errors.append((pred - actual) ** 2)

print("Mean Brier score:", sum(errors) / len(errors))
