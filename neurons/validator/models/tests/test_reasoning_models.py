from datetime import datetime

import pytest
from pydantic import ValidationError

from neurons.validator.models.reasoning import ReasoningModel


class TestReasoningModel:
    def test_create_minimal(self):
        model = ReasoningModel(
            event_id="evt001",
            miner_uid=1,
            miner_hotkey="hk1",
            reasoning="reasoning",
        )
        assert model.event_id == "evt001"
        assert model.miner_uid == 1
        assert model.miner_hotkey == "hk1"
        assert model.reasoning == "reasoning"

        # Defaults
        assert model.created_at is None
        assert model.updated_at is None
        assert model.exported is False

    def test_create_full(self):
        dt = datetime(2024, 1, 1, 12, 0)

        model = ReasoningModel(
            event_id="evt002",
            miner_uid=2,
            miner_hotkey="hk2",
            reasoning="reasoning",
            created_at=dt,
            updated_at=dt,
            exported=True,
        )

        assert model.event_id == "evt002"
        assert model.miner_uid == 2
        assert model.miner_hotkey == "hk2"
        assert model.reasoning == "reasoning"
        assert model.created_at == dt
        assert model.updated_at == dt
        assert model.exported is True

    @pytest.mark.parametrize(
        "exported_input, expected",
        [
            (1, True),
            (0, False),
        ],
    )
    def test_exported_int_to_bool(self, exported_input, expected):
        model = ReasoningModel(
            event_id="evt004",
            miner_uid=4,
            miner_hotkey="hk4",
            reasoning="reasoning",
            exported=exported_input,
        )
        assert model.exported is expected

    def test_exported_bool_passthrough(self):
        model = ReasoningModel(
            event_id="evt005",
            miner_uid=5,
            miner_hotkey="hk5",
            reasoning="reasoning",
            exported=True,
        )
        assert model.exported is True

    def test_invalid_data_type(self):
        # event_id must be a string.
        with pytest.raises(ValidationError):
            ReasoningModel(
                event_id=123,  # not a string
                miner_uid=7,
                miner_hotkey="hk7",
                reasoning="reasoning",
            )

        # miner_uid must be an int.
        with pytest.raises(ValidationError):
            ReasoningModel(
                event_id="evt006",
                miner_uid="not-an-int",
                miner_hotkey="hk8",
                reasoning="reasoning",
            )

    def test_primary_key_property(self):
        model = ReasoningModel(
            event_id="evt008",
            miner_uid=9,
            miner_hotkey="hk10",
            reasoning="reasoning",
        )

        # Expected primary key based on the ReasoningModel definition.
        assert model.primary_key == ["event_id", "miner_uid", "miner_hotkey"]
