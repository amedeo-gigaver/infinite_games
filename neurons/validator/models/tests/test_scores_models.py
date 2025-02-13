from datetime import datetime

import pytest
from pydantic import ValidationError

from neurons.validator.models.score import ScoresModel


class TestScoresModel:
    def test_create_minimal(self):
        model = ScoresModel(
            event_id="evt001",
            miner_uid=1,
            miner_hotkey="hk1",
            prediction=0.75,
            event_score=0.85,
            spec_version=1,
        )
        assert model.event_id == "evt001"
        assert model.miner_uid == 1
        assert model.miner_hotkey == "hk1"
        assert model.prediction == 0.75
        assert model.event_score == 0.85
        # Defaults
        assert model.metagraph_score is None
        assert model.other_data is None
        assert model.created_at is None
        assert model.processed is False
        assert model.exported is False

    def test_create_full(self):
        dt = datetime(2024, 1, 1, 12, 0)
        model = ScoresModel(
            event_id="evt002",
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.95,
            event_score=0.90,
            metagraph_score=0.88,
            other_data="extra",
            created_at=dt,
            spec_version=2,
            processed=True,
            exported=True,
        )
        assert model.event_id == "evt002"
        assert model.miner_uid == 2
        assert model.miner_hotkey == "hk2"
        assert model.prediction == 0.95
        assert model.event_score == 0.90
        assert model.metagraph_score == 0.88
        assert model.other_data == "extra"
        assert model.created_at == dt
        assert model.spec_version == 2
        assert model.processed is True
        assert model.exported is True

    @pytest.mark.parametrize(
        "processed_input, expected",
        [
            (1, True),
            (0, False),
        ],
    )
    def test_processed_int_to_bool(self, processed_input, expected):
        model = ScoresModel(
            event_id="evt003",
            miner_uid=3,
            miner_hotkey="hk3",
            prediction=0.80,
            event_score=0.82,
            spec_version=1,
            processed=processed_input,
        )
        assert model.processed is expected

    @pytest.mark.parametrize(
        "exported_input, expected",
        [
            (1, True),
            (0, False),
        ],
    )
    def test_exported_int_to_bool(self, exported_input, expected):
        model = ScoresModel(
            event_id="evt004",
            miner_uid=4,
            miner_hotkey="hk4",
            prediction=0.70,
            event_score=0.75,
            spec_version=1,
            exported=exported_input,
        )
        assert model.exported is expected

    def test_exported_bool_passthrough(self):
        model = ScoresModel(
            event_id="evt005",
            miner_uid=5,
            miner_hotkey="hk5",
            prediction=0.85,
            event_score=0.80,
            spec_version=1,
            exported=False,
        )
        assert model.exported is False

    def test_invalid_data_type(self):
        # event_id must be a string.
        with pytest.raises(ValidationError):
            ScoresModel(
                event_id=123,  # not a string
                miner_uid=7,
                miner_hotkey="hk7",
                prediction=0.85,
                event_score=0.80,
                spec_version=1,
            )
        # miner_uid must be an int.
        with pytest.raises(ValidationError):
            ScoresModel(
                event_id="evt006",
                miner_uid="not-an-int",
                miner_hotkey="hk8",
                prediction=0.75,
                event_score=0.70,
                spec_version=1,
            )
        # prediction must be a float.
        with pytest.raises(ValidationError):
            ScoresModel(
                event_id="evt007",
                miner_uid=8,
                miner_hotkey="hk9",
                prediction="not-a-float",
                event_score=0.65,
                spec_version=1,
            )

    def test_primary_key_property(self):
        model = ScoresModel(
            event_id="evt008",
            miner_uid=9,
            miner_hotkey="hk10",
            prediction=0.55,
            event_score=0.60,
            spec_version=1,
        )
        # Expected primary key based on the ScoresModel definition.
        assert model.primary_key == ["event_id", "miner_uid", "miner_hotkey"]
