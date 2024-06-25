
from bittensor.mock.wallet_mock import MockWallet
from bittensor.mock import MockSubtensor
from pytest import fixture

@fixture
def mock_network(monkeypatch):

    monkeypatch.setattr('infinite_games.base.neuron.get_wallet', lambda a: MockWallet())
    monkeypatch.setattr('infinite_games.base.neuron.get_subtensor', lambda a: MockSubtensor())