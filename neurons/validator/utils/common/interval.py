from datetime import datetime, timedelta, timezone

# The base time epoch for clustering intervals.
SCORING_REFERENCE_DATE = datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)

# Intervals are grouped in 4-hour blocks (240 minutes).
AGGREGATION_INTERVAL_LENGTH_MINUTES = 60 * 4

BLOCK_DURATION = 12  # 12 seconds block duration from bittensor


def minutes_since_epoch(dt: datetime) -> int:
    """Convert a given datetime to the 'minutes since the reference date'."""
    return int((dt - SCORING_REFERENCE_DATE).total_seconds()) // 60


def align_to_interval(minutes_since: int) -> int:
    """
    Align a given number of minutes_since_epoch down to
    the nearest AGGREGATION_INTERVAL_LENGTH_MINUTES boundary.
    """
    return minutes_since - (minutes_since % AGGREGATION_INTERVAL_LENGTH_MINUTES)


def get_interval_start_minutes():
    now = datetime.now(timezone.utc)

    mins_since_epoch = minutes_since_epoch(now)
    interval_start_minutes = align_to_interval(mins_since_epoch)

    return interval_start_minutes


def get_interval_iso_datetime(interval_start_minutes: int):
    return (SCORING_REFERENCE_DATE + timedelta(minutes=interval_start_minutes)).isoformat()
