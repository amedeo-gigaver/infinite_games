WITH last_rowid AS (
    -- Get the last ROWID for each miner in the last 2 days where processed = 1.
	SELECT
		miner_uid,
		miner_hotkey,
		MAX(ROWID) AS max_rowid
	FROM scores
	WHERE processed = 1
		AND created_at > datetime(CURRENT_TIMESTAMP, '-2 day')
	GROUP BY miner_uid, miner_hotkey
),
latest_metagraph_scores AS (
    -- Get the latest metagraph scores for each miner.
	SELECT
		lr.miner_uid,
		lr.miner_hotkey,
		s.metagraph_score
	FROM scores s
	JOIN last_rowid lr
		ON s.miner_uid = lr.miner_uid
		AND s.miner_hotkey = lr.miner_hotkey
		AND s.ROWID = lr.max_rowid
),
event_predictions AS (
    -- Get the latest predictions for the event from the top 10 miners.
	SELECT
        lms.miner_uid,
        lms.miner_hotkey,
        lms.metagraph_score,
        pr.interval_start_minutes,
        pr.interval_agg_prediction
	FROM latest_metagraph_scores lms
	INNER JOIN predictions pr
		ON pr.minerUid = lms.miner_uid
		AND pr.minerHotkey = lms.miner_hotkey
	WHERE pr.unique_event_id = :unique_event_id
		AND interval_start_minutes = :interval_start_minutes
	ORDER BY lms.metagraph_score DESC
	LIMIT 10
)
SELECT
	SUM(metagraph_score*interval_agg_prediction)/SUM(metagraph_score)
        AS weighted_average_prediction
FROM event_predictions;
