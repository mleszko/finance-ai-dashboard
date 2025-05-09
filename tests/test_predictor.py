import numpy as np
import pytest
import torch
from timeseries.predictor import prepare_sequences


def test_prepare_sequences_valid_input():
    data = np.arange(30)
    sequence_length = 10
    X, y = prepare_sequences(data, sequence_length)
    assert X.shape == (20, sequence_length, 1)
    assert y.shape == (20, 1)
    assert isinstance(X, torch.FloatTensor)
    assert isinstance(y, torch.FloatTensor)


def test_prepare_sequences_invalid_length():
    data = np.arange(10)
    sequence_length = 20
    with pytest.raises(ValueError):
        prepare_sequences(data, sequence_length)


def test_prepare_sequences_1d_data():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sequence_length = 5
    X, y = prepare_sequences(data, sequence_length)
    assert X.shape == (5, sequence_length, 1)
    assert y.shape == (5, 1)
    assert np.allclose(X[0], np.array([[1], [2], [3], [4], [5]]))
    assert np.allclose(y[0], np.array([6]))


def test_prepare_sequences_2d_data():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    sequence_length = 3
    X, y = prepare_sequences(data, sequence_length)
    assert X.shape == (5, sequence_length, 2)
    assert y.shape == (5, 2)
    assert np.allclose(X[0], np.array([[1, 2], [3, 4], [5, 6]]))
    assert np.allclose(y[0], np.array([7, 8]))


def test_prepare_sequences_single_step():
    data = np.arange(15).reshape(-1, 1)
    sequence_length = 1
    X, y = prepare_sequences(data, sequence_length)
    assert X.shape == (14, 1, 1)
    assert y.shape == (14, 1)


def test_prepare_sequences_edge_case_sequence_length_equals_data_length_plus_one():
    data = np.arange(21)
    sequence_length = 20
    with pytest.raises(ValueError):
        prepare_sequences(data, sequence_length)
