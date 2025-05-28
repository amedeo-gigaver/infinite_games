from datetime import datetime

import pytest
from pydantic import ValidationError

from neurons.validator.models.prediction import PredictionsModel


class TestPredictionsModel:
    def test_create_minimal(self):
        # Minimal required fields: unique_event_id, minerUid,
        # interval_start_minutes, interval_agg_prediction
        model = PredictionsModel(
            unique_event_id="event123",
            miner_uid=99,
            miner_hotkey="hotkeyXYZ",
            latest_prediction=0.5,
            interval_start_minutes=10,
            interval_agg_prediction=0.75,
        )
        assert model.unique_event_id == "event123"
        assert model.miner_uid == 99
        assert model.miner_hotkey == "hotkeyXYZ"
        assert model.latest_prediction == 0.5
        assert model.interval_start_minutes == 10
        assert model.interval_agg_prediction == 0.75

        # Defaults
        assert model.interval_count == 1
        assert model.submitted is None
        assert model.updated_at is None
        assert model.exported is False

    def test_create_full(self):
        model = PredictionsModel(
            unique_event_id="uniqueX",
            miner_uid=7,
            miner_hotkey="hotkey123",
            latest_prediction=0.1133,
            interval_start_minutes=100,
            interval_agg_prediction=1.23,
            interval_count=5,
            submitted=datetime(2024, 1, 1, 12, 30),
            updated_at=datetime(2025, 1, 1, 12, 30),
            exported=True,
        )
        assert model.unique_event_id == "uniqueX"
        assert model.miner_uid == 7
        assert model.miner_hotkey == "hotkey123"
        assert model.latest_prediction == 0.1133
        assert model.interval_start_minutes == 100
        assert model.interval_agg_prediction == 1.23
        assert model.interval_count == 5
        assert model.submitted == datetime(2024, 1, 1, 12, 30)
        assert model.updated_at == datetime(2025, 1, 1, 12, 30)
        assert model.exported is True

    def test_exported_int_to_bool(self):
        # If exported is an integer, it should become a bool
        model = PredictionsModel(
            unique_event_id="uniqueX",
            miner_uid=7,
            miner_hotkey="hotkey123",
            latest_prediction=0.1133,
            interval_start_minutes=100,
            interval_agg_prediction=1.23,
            exported=1,
        )
        assert model.exported is True

        model2 = PredictionsModel(
            unique_event_id="uniqueX",
            miner_uid=7,
            miner_hotkey="hotkey123",
            latest_prediction=0.1133,
            interval_start_minutes=100,
            interval_agg_prediction=1.23,
            exported=1,
        )
        assert model2.exported is True

    def test_exported_bool_passthrough(self):
        # If exported is already a boolean, no conversion needed
        model = PredictionsModel(
            unique_event_id="uniqueX",
            miner_uid=7,
            miner_hotkey="hotkey123",
            latest_prediction=0.1133,
            interval_start_minutes=100,
            interval_agg_prediction=1.23,
            exported=True,
        )
        assert model.exported is True

    def test_invalid_data_type(self):
        # interval_start_minutes must be an int
        with pytest.raises(ValidationError):
            PredictionsModel(
                unique_event_id="uniqueX",
                miner_uid=7,
                miner_hotkey="hotkey123",
                latest_prediction=0.1133,
                interval_start_minutes="not-an-int",
                interval_agg_prediction=1.23,
            )

        # interval_agg_prediction must be a float
        with pytest.raises(ValidationError):
            PredictionsModel(
                unique_event_id="uniqueX",
                miner_uid=7,
                miner_hotkey="hotkey123",
                latest_prediction=0.1133,
                interval_start_minutes=100,
                interval_agg_prediction="not-a-float",
            )

    def test_unique_constraints_property(self):
        model = PredictionsModel(
            unique_event_id="uniqueX",
            miner_uid=7,
            miner_hotkey="hotkey123",
            latest_prediction=0.1133,
            interval_start_minutes=100,
            interval_agg_prediction=1.23,
        )
        assert model.primary_key == [
            "unique_event_id",
            "miner_uid",
            "miner_hotkey",
            "interval_start_minutes",
        ]
