import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.models.prediction import PredictionExportedStatus
from infinite_games.sandbox.validator.tasks.export_predictions import ExportPredictions
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class TestExportPredictions:
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
    def export_predictions_task(self, db_operations: DatabaseOperations):
        api_client = IfGamesClient(env="test", logger=MagicMock(spec=AbstractLogger))

        return ExportPredictions(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            batch_size=1,
            validator_uid=11,
            validator_hotkey="validator_hotkey_test",
        )

    async def test_run(
        self,
        db_client: Client,
        db_operations: DatabaseOperations,
        export_predictions_task: ExportPredictions,
    ):
        """Test the run method when there are predictions to export."""

        # Mock API client
        export_predictions_task.api_client = AsyncMock(spec=IfGamesClient)

        events = [
            (
                "unique_event_id_1",
                "event_1",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                "status1",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
            (
                "unique_event_id_2",
                "event_2",
                "truncated_market2",
                "market_2",
                "desc2",
                "2024-12-02",
                "2024-12-03",
                "outcome2",
                "status2",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "unique_event_id_1",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "unique_event_id_2",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Act
        await export_predictions_task.run()

        # Assert
        assert export_predictions_task.api_client.post_predictions.call_count == 2

        mock_calls = export_predictions_task.api_client.post_predictions.mock_calls
        first_call = mock_calls[0]
        second_call = mock_calls[1]

        assert first_call == (
            "__call__",
            {
                "predictions": {
                    "events": None,
                    "submissions": [
                        {
                            "unique_event_id": "unique_event_id_1",
                            "provider_type": "market_1",
                            "prediction": "1",
                            "interval_start_minutes": 10,
                            "interval_agg_prediction": 1.0,
                            "interval_agg_count": 1,
                            "interval_datetime": "2024-01-01T00:10:00+00:00",
                            "miner_hotkey": "neuronHotkey_1",
                            "miner_uid": "neuronUid_1",
                            "validator_hotkey": "validator_hotkey_test",
                            "validator_uid": 11,
                            "title": None,
                            "outcome": None,
                        }
                    ],
                }
            },
        )
        assert second_call == (
            "__call__",
            {
                "predictions": {
                    "events": None,
                    "submissions": [
                        {
                            "unique_event_id": "unique_event_id_2",
                            "provider_type": "market_2",
                            "prediction": "1",
                            "interval_start_minutes": 10,
                            "interval_agg_prediction": 1.0,
                            "interval_agg_count": 1,
                            "interval_datetime": "2024-01-01T00:10:00+00:00",
                            "miner_hotkey": "neuronHotkey_2",
                            "miner_uid": "neuronUid_2",
                            "validator_hotkey": "validator_hotkey_test",
                            "validator_uid": 11,
                            "title": None,
                            "outcome": None,
                        }
                    ],
                }
            },
        )

        result = await db_client.many(
            """
                SELECT exported FROM predictions
            """
        )

        assert len(result) == 2
        assert result[0][0] == PredictionExportedStatus.EXPORTED
        assert result[1][0] == PredictionExportedStatus.EXPORTED

    async def test_run_no_predictions(self, export_predictions_task: ExportPredictions):
        # Mock API client
        export_predictions_task.api_client = AsyncMock(spec=IfGamesClient)

        # Act
        await export_predictions_task.run()

        # Assert
        export_predictions_task.api_client.post_predictions.assert_not_called()
