WITH reference_event AS (
    -- Get the reference ROWID for the specified event.
    SELECT MIN(ROWID) AS reference_row
    FROM scores
    WHERE event_id = :event_id
),
all_events AS (
    -- For each event, get the smallest ROWID (as a proxy for event order).
    SELECT event_id, MIN(ROWID) AS event_min_row
    FROM scores
    GROUP BY event_id
),
last_n_events AS (
    -- Select the last N events that occurred before the reference event.
    SELECT event_id
    FROM all_events
    WHERE event_min_row < (SELECT reference_row FROM reference_event)
    ORDER BY event_min_row DESC
    LIMIT :n_events
),
yes_stats AS (
    -- compute YES-rate in that window
    SELECT
        COUNT(1) AS n_window,
        COALESCE(SUM(CASE WHEN outcome = "1" THEN 1 ELSE 0 END), 0) AS yes_cnt
    FROM events
    WHERE event_id IN (SELECT event_id FROM last_n_events)
),
class_weights AS (
  -- Laplace-smoothed YES probability
  SELECT
    (yes_cnt + 1.0) * 1.0 / (n_window + 2.0) AS q_yes,
    n_window,
    yes_cnt
  FROM yes_stats
),
weights AS (
  SELECT
    1.0 / q_yes AS w1,
    1.0 / (1.0 - q_yes) AS w0
  FROM class_weights
),
weight_totals AS (
  SELECT
      /*  window_weight = YES* w1  +  NO * w0  */
      (yes_cnt * w1) +
      ((n_window - yes_cnt) * w0)     AS window_weight
  FROM class_weights CROSS JOIN weights
),
pre_aggregate AS (
    -- Aggregate weighted peer scores across the selected events.
    SELECT
        s.miner_uid,
        s.miner_hotkey,
        SUM(
            s.event_score *
            CASE WHEN ev.outcome = "1"
                    THEN (SELECT w1 FROM weights)
                    ELSE (SELECT w0 FROM weights)
            END
        ) AS sum_wpeer_score,
        SUM(
            CASE WHEN ev.outcome = "1"
                    THEN (SELECT w1 FROM weights)
                    ELSE (SELECT w0 FROM weights)
            END
        ) AS sum_weight,
        COUNT(event_score) AS count_peer_score
    FROM scores  AS s
    JOIN events  AS ev  ON ev.event_id = s.event_id
    WHERE s.event_id IN (SELECT event_id FROM last_n_events)
        AND ev.event_id IN (SELECT event_id FROM last_n_events)
    GROUP BY s.miner_uid, s.miner_hotkey
),
current_event AS (
    -- Get weighted scores for the current event.
    SELECT
        s.miner_uid,
        s.miner_hotkey,
        s.event_score *
            CASE WHEN ev.outcome = "1"
                THEN (SELECT w1 FROM weights)
                ELSE (SELECT w0 FROM weights)
            END AS weighted_score,
        CASE WHEN ev.outcome = "1"
            THEN (SELECT w1 FROM weights)
            ELSE (SELECT w0 FROM weights)
        END AS this_weight
    FROM scores  AS s
    JOIN events  AS ev  ON ev.event_id = s.event_id
    WHERE s.event_id = :event_id
      AND ev.event_id = :event_id
),
joined_data AS (
    -- Join current event scores with the aggregated scores.
    SELECT
        ce.miner_uid,
        ce.miner_hotkey,
        COALESCE(pa.sum_wpeer_score,0) + ce.weighted_score AS sum_wpeer_score,
        /* ------  fixed denominator  ------ */
        (SELECT window_weight FROM weight_totals) + ce.this_weight  AS sum_weight,
        1 + max(COALESCE(pa.count_peer_score, 0), :n_events) AS count_peer_score,
        1 + COALESCE(pa.count_peer_score, 0) AS true_count_peer_score
    FROM current_event ce
    LEFT JOIN pre_aggregate pa
        ON ce.miner_uid = pa.miner_uid
        AND ce.miner_hotkey = pa.miner_hotkey
),
avg_scores AS (
    -- Compute moving average and square max average peer scores.
    SELECT
        *,
        sum_wpeer_score / sum_weight AS avg_peer_score,
        -- sadly no power operator in sqlite
        (
            max(sum_wpeer_score / sum_weight, 0)
        ) * (
            max(sum_wpeer_score / sum_weight, 0)
        ) AS sqmax_avg_peer_score
    FROM joined_data
),
norm_base AS (
    -- Compute the normalization base: sum of squared average scores, mind div by 0
    SELECT max(sum(sqmax_avg_peer_score), 0.0000001) AS sum_sqmax_avg_peer_score
    FROM avg_scores
),
norm_scores AS (
    -- Normalize each squared metagraph score.
    SELECT
        *,
        sqmax_avg_peer_score / (
            SELECT sum_sqmax_avg_peer_score FROM norm_base
        ) AS metagraph_score
    FROM avg_scores
),
payload AS (
    -- Prepare the final payload with debug information in JSON.
    SELECT
        miner_uid,
        miner_hotkey,
        metagraph_score,
        json_object(
            'sum_weighted_peer_score', sum_wpeer_score,
            'sum_weight', sum_weight,
            'count_peer_score', count_peer_score,
            'true_count_peer_score', true_count_peer_score,
            'avg_peer_score', avg_peer_score,
            'sqmax_avg_peer_score', sqmax_avg_peer_score
        ) AS other_data
    FROM norm_scores
)
UPDATE scores
SET
    metagraph_score = (
        SELECT metagraph_score FROM payload
        WHERE miner_uid = scores.miner_uid
        AND miner_hotkey = scores.miner_hotkey
    ),
    other_data = (
        SELECT other_data FROM payload
        WHERE miner_uid = scores.miner_uid
        AND miner_hotkey = scores.miner_hotkey
    ),
    processed = 1
WHERE event_id = :event_id
;