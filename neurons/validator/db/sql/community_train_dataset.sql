WITH ranked_events AS (
    SELECT
        event_id,
        ROW_NUMBER() OVER(
            ORDER BY MIN(ROWID) DESC
        ) AS event_rank,
        datetime(
            CAST(strftime('%s', MIN(created_at)) / 14400 AS INTEGER) * 14400,
            'unixepoch'
        ) AS scoring_batch
    FROM scores
    WHERE created_at >= datetime(CURRENT_TIMESTAMP, '-15 day')
    GROUP BY event_id
  ),

  batch_event_ranks AS (
    SELECT
      event_id,
      event_rank,
      scoring_batch,
      DENSE_RANK() OVER (
        ORDER BY scoring_batch DESC
      ) AS batch_rank,
      ROW_NUMBER() OVER (
        PARTITION BY scoring_batch
        ORDER BY event_rank
      ) AS event_batch_rank
    FROM ranked_events
  ),

  raw_data AS (
    SELECT
      se.miner_uid,
      se.miner_hotkey,
      ber.event_rank,
      se.event_id,
      ber.scoring_batch,
      ber.batch_rank,
      ber.event_batch_rank,
      se.prediction AS agg_prediction,
      CAST(ev.outcome AS INTEGER) AS outcome,
      se.metagraph_score,
      datetime(ber.scoring_batch, '-4 hours') AS prev_batch
    FROM scores AS se
    JOIN events AS ev ON se.event_id = ev.event_id
    JOIN batch_event_ranks AS ber ON se.event_id = ber.event_id
    WHERE se.created_at >= datetime(CURRENT_TIMESTAMP, '-15 day')
        AND ev.status = '3'
        AND ev.resolved_at >= datetime(CURRENT_TIMESTAMP, '-16 day')
  ),

  raw_ranked AS (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY event_rank
        ORDER BY metagraph_score DESC,
                 miner_uid DESC -- tie-break
      ) AS miner_rank
    FROM raw_data
  ),

  prev_batch_tops AS (
    /* pick out each miner’s rank on the single event_batch_rank=1” row
       (i.e. the most-recent event) so we can join it back in */
    SELECT
      miner_uid,
      miner_hotkey,
      miner_rank AS prev_batch_miner_rank,
      scoring_batch,
      metagraph_score AS prev_metagraph_score
    FROM raw_ranked
    WHERE event_batch_rank = 1
  )

SELECT
    rr.miner_uid,
    rr.miner_hotkey,
    rr.miner_rank,
    COALESCE(p.prev_batch_miner_rank, 256) AS prev_batch_miner_rank,
    COALESCE(p.prev_metagraph_score, 0.000001) AS prev_metagraph_score,
    rr.event_id,
    rr.event_rank,
    rr.scoring_batch,
    rr.batch_rank,
    rr.event_batch_rank,
    rr.outcome,
    rr.agg_prediction,
    rr.metagraph_score
FROM raw_ranked AS rr
LEFT JOIN prev_batch_tops AS p
  ON p.miner_uid = rr.miner_uid
  AND p.miner_hotkey = rr.miner_hotkey
  AND p.scoring_batch = rr.prev_batch
ORDER BY
  rr.miner_uid,
  rr.scoring_batch DESC,
  rr.event_rank;
