import numpy as np
from sklearn.isotonic import IsotonicRegression
from scripts.backtest_custom import data  # adapt to return lists

preds = np.array([d["pred"] for d in data])
actuals = np.array([d["actual"] for d in data])
iso = IsotonicRegression(out_of_bounds="clip").fit(preds, actuals)
calibrated = iso.transform(preds)
print("Calibrated Brier score:", np.mean((calibrated - actuals)**2))
