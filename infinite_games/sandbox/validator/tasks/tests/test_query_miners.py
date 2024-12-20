import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from bittensor.core.dendrite import DendriteMixin
from bittensor.core.metagraph import MetagraphMixin

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.models.events_prediction_synapse import (
    EventsPredictionSynapse,
)
from infinite_games.sandbox.validator.tasks.query_miners import QueryMiners
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class TestQueryMiners:
    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=AbstractLogger)

        db_client = Client(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    def db_operations(self, db_client: Client):
        return DatabaseOperations(db_client=db_client)

    @pytest.fixture
    def query_miners_task(self, db_operations: DatabaseOperations):
        dendrite = MagicMock(spec=DendriteMixin)
        metagraph = MagicMock(spec=MetagraphMixin)

        return QueryMiners(
            interval_seconds=60.0,
            db_operations=db_operations,
            dendrite=dendrite,
            metagraph=metagraph,
        )

    def test_get_axons(
        self,
        query_miners_task: QueryMiners,
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = [1, 2, 3]  # UIDs in the metagraph
        query_miners_task.metagraph.axons = {
            1: MagicMock(is_serving=True),  # Valid axon
            2: MagicMock(is_serving=False),  # Not serving
            3: None,  # No axon at this UID
        }

        # Call the method
        result = query_miners_task.get_axons()

        # Assertions
        # Only UID 1 should be included because it's serving
        assert result == {1: query_miners_task.metagraph.axons[1]}

    def test_get_axons_empty_metagraph(
        self,
        query_miners_task: QueryMiners,
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = []  # No UIDs in the metagraph

        # Call the method
        result = query_miners_task.get_axons()

        # Assertions
        # No axons should be included
        assert result == {}

    @pytest.mark.parametrize(
        "test_time,expected",
        [
            # Test exact epoch start
            (datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 0),
            # Test multiple days later
            (datetime(2025, 1, 3, 4, 0, 0, tzinfo=timezone.utc), 530160),
            # Test with non-zero minutes and seconds (should round down)
            (datetime(2024, 1, 1, 4, 45, 30, tzinfo=timezone.utc), 240),
        ],
    )
    def test_get_interval_start_minutes(self, query_miners_task: QueryMiners, test_time, expected):
        with patch("infinite_games.sandbox.validator.tasks.query_miners.datetime") as mock_datetime:
            # Configure mock to return our test time when now() is called
            mock_datetime.now.return_value = test_time

            result = query_miners_task.get_interval_start_minutes()

            assert result == expected

    def test_make_predictions_synapse(self, query_miners_task: QueryMiners):
        events_from_db = [
            (
                "event1",  # event_id
                "ifgames",  # market_type
                "Test match",  # description
                "2012-12-02T14:30:00+00:00",  # cutoff
                "2012-12-02T14:30:00+00:00",  # resolve_date
                "2012-12-03T14:30:00+00:00",  # end_date
                {
                    "market_type": "acled",
                },  # metadata
            ),
            (
                "event2",  # event_id
                "ifgames",  # market_type
                "Test match 2",  # description
                "2012-12-02T14:30:00+00:00",  # cutoff
                "2012-12-02T14:30:00+00:00",  # resolve_date
                "2012-12-03T14:30:00+00:00",  # end_date
                {
                    "market_type": "azuro",
                },  # metadata
            ),
        ]

        synapse = query_miners_task.make_predictions_synapse(events=events_from_db)

        assert isinstance(synapse, EventsPredictionSynapse)
        assert len(synapse.events) == 2

        # Assert event 1
        expected_key = "acled-event1"

        assert expected_key in synapse.events

        event_data = synapse.events[expected_key]

        assert event_data["event_id"] == "event1"
        assert event_data["market_type"] == "acled"
        assert event_data["description"] == "Test match"
        assert event_data["cutoff"] == 1354458600
        assert event_data["probability"] is None
        assert event_data["miner_answered"] is False
        assert event_data["starts"] is None
        assert event_data["resolve_date"] == 1354458600
        assert event_data["end_date"] == 1354545000

        # Assert event 2
        expected_key = "azuro-event2"

        assert expected_key in synapse.events

        event_data = synapse.events[expected_key]

        assert event_data["event_id"] == "event2"
        assert event_data["market_type"] == "azuro"
        assert event_data["description"] == "Test match 2"
        assert event_data["cutoff"] == 1354458600
        assert event_data["probability"] is None
        assert event_data["miner_answered"] is False
        assert event_data["starts"] == 1354458600
        assert event_data["resolve_date"] == 1354458600
        assert event_data["end_date"] == 1354545000

    def test_make_predictions_synapse_empty_events(self, query_miners_task: QueryMiners):
        """Test handling of empty events"""
        events = []

        result = query_miners_task.make_predictions_synapse(events=events)

        assert isinstance(result, EventsPredictionSynapse)
        assert len(result.events) == 0

    def test_parse_neuron_predictions(self, query_miners_task: QueryMiners):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = [1, 2]
        query_miners_task.metagraph.axons = {
            1: MagicMock(hotkey="hotkey1"),
            2: MagicMock(hotkey="hotkey2"),
        }

        interval_start_minutes = 12345
        block = 54321
        uid = 2

        neuron_predictions = EventsPredictionSynapse(
            events={
                "acled-event1": {
                    "event_id": "event1",
                    "market_type": "acled",
                    "probability": 0.5,
                    "miner_answered": True,
                    "description": "Test match",
                    "cutoff": 1354458600,
                    "starts": None,
                    "resolve_date": 1354458600,
                    "end_date": 1354545000,
                },
                "azuro-event2": {
                    "event_id": "event2",
                    "market_type": "azuro",
                    "probability": 0.75,
                    "miner_answered": False,
                    "description": "Test match 2",
                    "cutoff": 1354458600,
                    "starts": 1354458600,
                    "resolve_date": 1354458600,
                    "end_date": 1354545000,
                },
            }
        )

        result = query_miners_task.parse_neuron_predictions(
            block=block,
            interval_start_minutes=interval_start_minutes,
            uid=uid,
            neuron_predictions=neuron_predictions,
        )

        assert result == [
            (
                "acled-event1",
                query_miners_task.metagraph.axons[uid].hotkey,
                uid,
                0.5,
                interval_start_minutes,
                0.5,
                block,
                0.5,
            ),
            (
                "azuro-event2",
                query_miners_task.metagraph.axons[uid].hotkey,
                uid,
                0.75,
                interval_start_minutes,
                0.75,
                block,
                0.75,
            ),
        ]
