from pyneurode.utils import pos_bin_data, get_binned_mean
import pytest
import numpy as np 

@pytest.fixture
def pos():
    return np.array([1,2,3,1,2,3])

@pytest.fixture
def x():
    return np.array([1,2,3,4,5,6])


def test_pos_bin_data(pos,x):
    binned_idx,bins = pos_bin_data(pos,3)
    expected = np.array([0,1,2,0,1,2])
    assert np.allclose(binned_idx,expected)

def test_get_binned_mean(pos,x):
    binned_idx,bins = pos_bin_data(pos,3)

    mean_x = get_binned_mean(binned_idx,bins,x)
    expected = np.array([2.5,3.5,4.5])

    assert np.allclose(mean_x,expected)
