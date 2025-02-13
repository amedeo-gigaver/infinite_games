from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from neurons.validator.utils.common.interval import (
    align_to_interval,
    get_interval_iso_datetime,
    get_interval_start_minutes,
    minutes_since_epoch,
    to_utc,
)


class TestInterval:
    def test_to_utc(self):
        assert to_utc(datetime(2024, 12, 27, 0, 1, 0, 0)) == datetime(
            2024, 12, 27, 0, 1, 0, 0, timezone.utc
        )
        assert to_utc(datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc)) == datetime(
            2024, 12, 27, 0, 1, 0, 0, timezone.utc
        )

        cet = timezone(timedelta(hours=1))
        dt_cet = datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc).astimezone(cet)
        assert dt_cet == datetime(2024, 12, 27, 1, 1, 0, 0, cet)
        assert to_utc(dt_cet) == datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc)

    def test_minutes_since_epoch(self):
        assert minutes_since_epoch(datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc)) == 519841

    def test_align_to_interval(self):
        assert align_to_interval(519841) == 519840
        assert align_to_interval(520079) == 519840
        assert align_to_interval(520080) == 520080

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
    def test_get_interval_start_minutes(self, test_time, expected):
        with patch("neurons.validator.utils.common.interval.datetime") as mock_datetime:
            # Configure mock to return our test time when now() is called
            mock_datetime.now.return_value = test_time

            result = get_interval_start_minutes()

            assert result == expected

    def test_get_interval_iso_datetime(self):
        interval_start_minutes = 500000
        expected = "2024-12-13T05:20:00+00:00"

        result = get_interval_iso_datetime(interval_start_minutes)

        assert result == expected
