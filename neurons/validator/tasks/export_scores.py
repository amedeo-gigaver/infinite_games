import json

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.backend_models import MinerEventResult, MinerEventResultItems
from neurons.validator.models.event import EventsModel
from neurons.validator.models.score import ScoresModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class ExportScores(AbstractTask):
    interval: float
    page_size: int
    db_operations: DatabaseOperations
    api_client: IfGamesClient
    logger: InfiniteGamesLogger
    validator_uid: int
    validator_hotkey: str

    def __init__(
        self,
        interval_seconds: float,
        page_size: int,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
        logger: InfiniteGamesLogger,
        validator_uid: int,
        validator_hotkey: str,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.page_size = page_size
        self.db_operations = db_operations
        self.api_client = api_client
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey

        self.errors_count = 0
        self.logger = logger

    @property
    def name(self):
        return "export-scores"

    @property
    def interval_seconds(self):
        return self.interval

    def prepare_scores_payload(self, event: EventsModel, scores: list[ScoresModel]) -> list[dict]:
        results = []
        failures = 0
        # event.metadata guaranteed non null by the pydantic EventsModel
        event_market_type = json.loads(event.metadata).get("market_type", event.market_type)
        metadata = json.loads(event.metadata)
        for score in scores:
            try:
                score_metadata = metadata.copy()
                if score.other_data:
                    score_metadata["other_data"] = json.loads(score.other_data)
                # override the spec version to be at least 1039 - peer scoring start
                # also backend expects a string
                backend_spec_version = str(max(score.spec_version, 1039))
                result = MinerEventResult(
                    event_id=score.event_id,  # awful: backend reconstructs unique_event_id
                    provider_type=event_market_type,
                    title=event.description[:50],  # as in the original code
                    description=event.description,
                    category="event",  # as in the original code
                    start_date=event.starts.isoformat() if event.starts else None,
                    end_date=event.resolve_date.isoformat() if event.resolve_date else None,
                    resolve_date=event.resolve_date.isoformat() if event.resolve_date else None,
                    settle_date=event.cutoff.isoformat(),  # as in the original code
                    prediction=score.prediction,
                    answer=float(event.outcome),
                    miner_hotkey=score.miner_hotkey,
                    miner_uid=score.miner_uid,
                    miner_score=score.event_score,
                    miner_effective_score=score.metagraph_score,
                    validator_hotkey=self.validator_hotkey,
                    validator_uid=self.validator_uid,
                    metadata=score_metadata,
                    spec_version=backend_spec_version,
                    registered_date=event.registered_date.isoformat(),
                    scored_at=score.created_at.isoformat(),
                )
                results.append(result)
            except Exception:
                self.errors_count += 1
                failures += 1
                if failures < 5:  # prevent spamming the logs
                    self.logger.exception(
                        "Failed to parse a score payload.",
                        extra={"event": event.model_dump(), "score": score.model_dump()},
                    )
        if not results:
            return None

        try:
            payload = MinerEventResultItems(results=results).model_dump_json()
            return json.loads(payload)
        except Exception:
            self.errors_count += 1
            self.logger.exception(
                "Failed to model_dump() scores payload.", extra={"event_id": event.event_id}
            )
            return None

    async def export_scores_to_backend(self, payload: list[dict]):
        await self.api_client.post_scores(scores=payload)
        self.logger.debug(
            "Exported scores.",
            extra={
                "event_id": payload["results"][0]["event_id"],
                "n_scores": len(payload["results"]),
            },
        )

    async def run(self):
        scored_events = await self.db_operations.get_peer_scored_events_for_export(
            max_events=self.page_size
        )
        if not scored_events:
            self.logger.debug("No peer scored events to export scores.")
        else:
            self.logger.debug(
                "Found peer scored events to export scores.",
                extra={"n_events": len(scored_events)},
            )

            for event in scored_events:
                scores = await self.db_operations.get_peer_scores_for_export(
                    event_id=event.event_id
                )
                if not scores:
                    self.errors_count += 1
                    self.logger.warning(
                        "No peer scores found for event.",
                        extra={"event_id": event.event_id},
                    )
                    continue

                payload = self.prepare_scores_payload(event=event, scores=scores)
                if not payload:
                    self.errors_count += 1
                    self.logger.warning(
                        "Failed to prepare scores payload.",
                        extra={"event_id": event.event_id},
                    )
                    continue

                try:
                    await self.export_scores_to_backend(payload)
                except Exception:
                    self.errors_count += 1
                    self.logger.exception(
                        "Failed to export scores.",
                        extra={"event_id": event.event_id},
                    )
                    continue

                await self.db_operations.mark_peer_scores_as_exported(event_id=event.event_id)

                await self.db_operations.mark_event_as_exported(
                    unique_event_id=event.unique_event_id
                )

        self.logger.debug(
            "Export scores task completed.",
            extra={"errors_count": self.errors_count},
        )

        self.errors_count = 0
