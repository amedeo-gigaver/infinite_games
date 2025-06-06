-- TODO: merge this query with latest_predictions_events.sql
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
events_predictions AS (
    -- Get the latest predictions for the events
	SELECT
		pr.unique_event_id,
        pr.interval_agg_prediction,
        lms.metagraph_score,
		-- Rank them by metagraph_score per event id
		ROW_NUMBER() OVER (PARTITION BY pr.unique_event_id ORDER BY lms.metagraph_score DESC) AS rn
	FROM latest_metagraph_scores lms
	INNER JOIN predictions pr
		ON pr.miner_uid = lms.miner_uid
		AND pr.miner_hotkey = lms.miner_hotkey
	WHERE
		pr.unique_event_id IN (:unique_event_ids)
		AND interval_start_minutes = :interval_start_minutes
)
SELECT
	unique_event_id,
	rn AS prev_batch_miner_rank,
	metagraph_score,
	interval_agg_prediction AS agg_prediction
FROM events_predictions
WHERE rn <= :top_n_ranks;