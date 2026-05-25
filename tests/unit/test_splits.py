"""Tests for the SplitFractions configuration object.

The Spark-backed temporal_split itself is exercised by an integration
test under tests/integration/ where a SparkSession is available; here
we only check the validation contract of the SplitFractions class.
"""

from __future__ import annotations

import pytest

from gamereco.etl.splits import SplitFractions


def test_default_fractions_sum_to_less_than_one() -> None:
    f = SplitFractions()
    assert 0 < f.val < 1
    assert 0 < f.test < 1
    assert f.val + f.test < 1


def test_custom_fractions_valid() -> None:
    f = SplitFractions(val=0.2, test=0.1)
    assert f.val == 0.2
    assert f.test == 0.1


def test_invalid_val_zero_rejected() -> None:
    with pytest.raises(ValueError):
        SplitFractions(val=0.0, test=0.1)


def test_invalid_test_one_rejected() -> None:
    with pytest.raises(ValueError):
        SplitFractions(val=0.1, test=1.0)


def test_invalid_sum_rejected() -> None:
    with pytest.raises(ValueError):
        SplitFractions(val=0.6, test=0.5)
