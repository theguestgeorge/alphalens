"""Next-day direction models: chronological train/test split, evaluation and forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

RANDOM_STATE = 42
TEST_FRACTION = 0.3
MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 20
MODEL_NAMES = ("Logistic regression", "Random forest")


class InsufficientDataError(Exception):
    """Raised when there is not enough history to train or evaluate a model. The message is shown to the user."""


@dataclass
class Evaluation:
    """A model trained on the earlier rows and its predictions for the later, unseen rows."""

    model: Pipeline
    train: pd.DataFrame
    test: pd.DataFrame  # test rows with added prob_up and prediction columns
    accuracy: float
    baseline_accuracy: float  # accuracy of always predicting "up" on the test rows


@dataclass
class Forecast:
    as_of: pd.Timestamp  # date of the latest close the forecast is based on
    prob_up: float  # model-estimated probability that the next close is higher
    training_rows: int


def make_model(name: str) -> Pipeline:
    """Create an untrained model. Scaling sits inside the pipeline so it is fitted on training data only."""
    if name == "Logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "Random forest":
        forest = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE)
        return make_pipeline(forest)
    raise ValueError(f"Unknown model: {name}")


def chronological_split(
    labelled: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: the earliest rows train the model and the latest rows test it. No shuffling."""
    if not labelled.index.is_monotonic_increasing:
        raise ValueError("Rows must be sorted by date before splitting.")
    n_test = round(len(labelled) * test_fraction)
    n_train = len(labelled) - n_test
    if n_train < MIN_TRAIN_ROWS or n_test < MIN_TEST_ROWS:
        raise InsufficientDataError(
            f"Only {len(labelled)} usable trading days are available. At least "
            f"{MIN_TRAIN_ROWS} training and {MIN_TEST_ROWS} test days are needed. Choose a longer history."
        )
    return labelled.iloc[:n_train], labelled.iloc[n_train:]


def _fit(model_name: str, rows: pd.DataFrame) -> Pipeline:
    if rows["target"].nunique() < 2:
        raise InsufficientDataError(
            "The training data contains only up days or only down days, so a direction model cannot be trained."
        )
    model = make_model(model_name)
    model.fit(rows[FEATURE_COLUMNS], rows["target"].astype(int))
    return model


def _prob_up(model: Pipeline, rows: pd.DataFrame) -> pd.Series:
    up_column = list(model.classes_).index(1)
    return pd.Series(model.predict_proba(rows[FEATURE_COLUMNS])[:, up_column], index=rows.index)


def evaluate_model(dataset: pd.DataFrame, model_name: str) -> Evaluation:
    """Train on the earlier part of the history and predict the later part, which the model never sees."""
    labelled = dataset.dropna(subset=["target"])
    train, test = chronological_split(labelled)
    model = _fit(model_name, train)

    test = test.copy()
    test["prob_up"] = _prob_up(model, test)
    test["prediction"] = (test["prob_up"] > 0.5).astype(int)

    accuracy = float((test["prediction"] == test["target"]).mean())
    baseline_accuracy = float(test["target"].mean())
    return Evaluation(model, train, test, accuracy, baseline_accuracy)


def forecast_next_day(dataset: pd.DataFrame, model_name: str) -> Forecast:
    """Refit the model on all labelled history and estimate the direction of the next close.

    The latest row has features but no target, because its next close has not happened yet.
    """
    labelled = dataset.dropna(subset=["target"])
    if len(labelled) < MIN_TRAIN_ROWS:
        raise InsufficientDataError(
            f"Only {len(labelled)} usable trading days are available; at least {MIN_TRAIN_ROWS} are needed."
        )
    model = _fit(model_name, labelled)
    latest = dataset.iloc[[-1]]
    return Forecast(as_of=latest.index[0], prob_up=float(_prob_up(model, latest).iloc[0]), training_rows=len(labelled))
