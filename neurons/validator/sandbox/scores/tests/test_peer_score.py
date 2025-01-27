import pytest

from neurons.validator.sandbox.scores.peer_score import PredictionData, Scorer


class TestPredictionData:
    def test_valid_prediction_input(self):
        # Valid inputs should not raise any exceptions

        prediction = PredictionData(probability=0.8, outcome=1)

        assert prediction.probability == 0.8
        assert prediction.outcome == 1

    def test_invalid_input_probability_zero(self):
        with pytest.raises(ValueError, match="Prediction probability must be between \\+0 and 1"):
            PredictionData(probability=0.0, outcome=1)

    def test_invalid_input_probability_above_one(self):
        with pytest.raises(ValueError, match="Prediction probability must be between \\+0 and 1"):
            PredictionData(probability=1.1, outcome=1)

    def test_invalid_input_outcome(self):
        with pytest.raises(ValueError, match="Prediction outcome must be either 0 or 1"):
            PredictionData(probability=0.8, outcome=2)


class TestScorerLogScore:
    def test_invalid_prediction(self):
        scorer = Scorer()

        with pytest.raises(ValueError, match="Prediction required"):
            scorer.log_score(None, outcome=1)

    def test_invalid_outcome(self):
        scorer = Scorer()

        prediction = PredictionData(probability=0.8, outcome=1)

        with pytest.raises(ValueError, match="Invalid outcome value"):
            scorer.log_score(prediction, outcome=2)

    def test_log_score_correct_result(self):
        scorer = Scorer()

        # Case 1
        prediction = PredictionData(probability=0.99, outcome=1)

        score = scorer.log_score(prediction, outcome=1)
        assert score == pytest.approx(-0.01, abs=0.01)

        score = scorer.log_score(prediction, outcome=0)
        assert score == pytest.approx(-4.6, abs=0.01)

        # Case 2
        prediction = PredictionData(probability=0.999, outcome=1)

        score = scorer.log_score(prediction, outcome=1)
        assert score == pytest.approx(-0.001, abs=0.01)

        score = scorer.log_score(prediction, outcome=0)
        assert score == pytest.approx(-6.9, abs=0.01)

    def test_log_score_edge_cases(self):
        scorer = Scorer()

        # Max score
        score = scorer.log_score(PredictionData(probability=1, outcome=1), outcome=1)
        assert score == 0

        # Min score
        # score = scorer.log_score(PredictionData(probability=1, outcome=1), outcome=0)
        # assert score == 0
        # TODO


class TestScorerPeerScore:
    def test_log_score_correct_result2(self):
        scorer = Scorer()
        prediction = PredictionData(probability=0.8, outcome=1)

        print(scorer.peer_score([(1, prediction), (1, prediction)], 1))
