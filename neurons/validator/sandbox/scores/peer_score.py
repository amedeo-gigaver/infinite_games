import math
from dataclasses import dataclass
from typing import List, Tuple

MinerID = int


@dataclass
class PredictionData:
    probability: float  # Prediction probability assigned to the prediction (0, 1]
    outcome: int  # Prediction outcome (0 or 1)

    def __post_init__(self):
        # Guard for log score of 0 probability = - infinite
        if not (0.0 < self.probability <= 1.0):
            raise ValueError(
                f"Prediction probability must be between +0 and 1. Got {self.probability}."
            )

        if self.outcome not in {0, 1}:
            raise ValueError(f"Prediction outcome must be either 0 or 1. Got {self.outcome}.")


class Scorer:
    def __init__(self):
        pass

    def log_score(self, prediction: PredictionData, outcome: int) -> float:
        """
        Calculate the log score for a prediction.
        """

        # Validation
        if prediction is None:
            raise ValueError("Prediction required")

        if outcome not in {0, 1}:
            raise ValueError(f"Invalid outcome value {outcome}")

        prediction_probability = prediction.probability
        prediction_outcome = prediction.outcome

        if prediction_outcome == outcome:
            return math.log(prediction_probability)

        return math.log(1 - prediction_probability)

    def peer_score(
        self, predictions: List[Tuple[MinerID, PredictionData]], outcome: int
    ) -> List[Tuple[MinerID, float]]:
        # Assumes unique miners predictions passed
        miners_count = len(predictions)

        if miners_count < 2:
            raise ValueError(f"More than 1 miners required, len {miners_count}")

        # Calculate log scores
        log_scores_by_miner = {}

        for miner_id, prediction in predictions:
            log_score = self.log_score(prediction, outcome)

            log_scores_by_miner[miner_id] = log_score

        # Calculate peer scores
        peer_scores_by_miner = {}

        for miner_id, miner_log_score in log_scores_by_miner.items():
            sum = 0

            for other_miner_id, other_miner_log_score in log_scores_by_miner.items():
                if miner_id != other_miner_id:
                    sum += miner_log_score - other_miner_log_score

            peer_scores_by_miner[miner_id] = 100 * sum / (miners_count - 1)

        # Return a list of tuples (MinerID, Peer Score)
        return list(peer_scores_by_miner.items())
