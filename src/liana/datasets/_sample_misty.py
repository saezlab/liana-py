import numpy as np
import pandas as pd


def _sample_interactions(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = ["target", "predictor", "view", "importances"]
    interactions = pd.DataFrame(columns=columns)
    interactions["target"] = np.repeat(["a", "b", "c"], 3)
    interactions["predictor"] = np.tile(["x", "y", "z"], 3)
    interactions["view"] = np.repeat(["intra", "inter", "extra"], 3)
    interactions["importances"] = rng.random(9)

    return interactions


def _sample_target_metrics(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = ["target", "intra_R2", "multi_R2", "gain_R2", "intra", "extra"]
    target_metrics = pd.DataFrame(columns=columns)
    target_metrics["target"] = ["a", "b", "c"]
    target_metrics["intra_R2"] = rng.random(3)
    target_metrics["multi_R2"] = rng.random(3)
    target_metrics["gain_R2"] = rng.random(3)
    target_metrics["intra"] = rng.random(3)
    target_metrics["extra"] = rng.random(3)

    return target_metrics
