import os

os.environ['NEUTUID'] = "155"
from bittensor.mock.wallet_mock import MockWallet
from bittensor.mock import MockSubtensor
from pytest import fixture


@fixture
def mock_network(monkeypatch):
    w = MockWallet()
    s = MockSubtensor()
    s.create_subnet(155)
    monkeypatch.setattr('infinite_games.base.neuron.get_wallet', lambda a: w)
    monkeypatch.setattr('infinite_games.base.neuron.get_subtensor', lambda a: s)

    yield w, s
