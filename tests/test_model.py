import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, build_dataset
from model import (
    MODEL_NAMES,
    TEST_FRACTION,
    InsufficientDataError,
    chronological_split,
    evaluate_model,
    forecast_next_day,
    make_model,
)


@pytest.fixture
def dataset(make_prices):
    return build_dataset(make_prices(400))


def test_chronological_split_keeps_time_order(dataset):
    labelled = dataset.dropna(subset=["target"])
    train, test = chronological_split(labelled)

    assert train.index.max() < test.index.min()
    assert len(test) == round(len(labelled) * TEST_FRACTION)
    # Nothing is dropped, duplicated or reordered
    pd.testing.assert_frame_equal(pd.concat([train, test]), labelled)


def test_chronological_split_rejects_unsorted_rows(dataset):
    with pytest.raises(ValueError):
        chronological_split(dataset.dropna(subset=["target"]).iloc[::-1])


def test_chronological_split_requires_enough_history(make_prices):
    labelled = build_dataset(make_prices(100)).dropna(subset=["target"])
    with pytest.raises(InsufficientDataError):
        chronological_split(labelled)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_evaluation_predicts_only_later_unseen_rows(dataset, model_name):
    result = evaluate_model(dataset, model_name)

    assert result.train.index.max() < result.test.index.min()
    assert result.test["prob_up"].between(0, 1).all()
    assert set(result.test["prediction"]) <= {0, 1}
    # The latest row has no known outcome, so it is in neither set
    assert dataset.index[-1] not in result.train.index
    assert dataset.index[-1] not in result.test.index
    assert result.accuracy == pytest.approx((result.test["prediction"] == result.test["target"]).mean())
    assert result.baseline_accuracy == pytest.approx(result.test["target"].mean())


def test_scaler_is_fitted_on_training_rows_only(dataset):
    result = evaluate_model(dataset, "Logistic regression")
    scaler = result.model.named_steps["standardscaler"]

    np.testing.assert_allclose(scaler.mean_, result.train[FEATURE_COLUMNS].mean().to_numpy())


def test_test_period_outcomes_do_not_influence_the_model(dataset):
    _, test = chronological_split(dataset.dropna(subset=["target"]))
    altered = dataset.copy()
    altered.loc[test.index, "target"] = 1 - altered.loc[test.index, "target"]

    original = evaluate_model(dataset, "Random forest").test["prob_up"]
    flipped = evaluate_model(altered, "Random forest").test["prob_up"]

    pd.testing.assert_series_equal(original, flipped)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_models_are_reproducible(dataset, model_name):
    first = evaluate_model(dataset, model_name).test["prob_up"]
    second = evaluate_model(dataset, model_name).test["prob_up"]
    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_models_learn_a_predictable_pattern(model_name):
    # Prices alternate +2% and -1%, so the latest daily return fully determines the next direction
    steps = np.where(np.arange(400) % 2 == 0, 1.02, 0.99)
    prices = pd.DataFrame({"Close": 100 * np.cumprod(steps)}, index=pd.bdate_range("2020-01-01", periods=400))

    result = evaluate_model(build_dataset(prices), model_name)

    assert result.accuracy > 0.95


def test_training_data_with_one_class_is_rejected(dataset):
    one_class = dataset.copy()
    one_class["target"] = one_class["target"].where(one_class["target"].isna(), 1.0)
    with pytest.raises(InsufficientDataError):
        evaluate_model(one_class, "Logistic regression")


def test_forecast_uses_latest_row_and_all_labelled_history(dataset):
    forecast = forecast_next_day(dataset, "Logistic regression")

    assert forecast.as_of == dataset.index[-1]
    assert forecast.training_rows == len(dataset) - 1
    assert 0 <= forecast.prob_up <= 1


def test_forecast_requires_enough_history(make_prices):
    with pytest.raises(InsufficientDataError):
        forecast_next_day(build_dataset(make_prices(60)), "Logistic regression")


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        make_model("Neural network")
