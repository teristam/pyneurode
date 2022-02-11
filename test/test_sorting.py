from online_sorter import simpleDownSample
import numpy as np 
from pyneurode.spike_sorter import *
import pytest

@pytest.fixture
def df_spikes():
    return pd.read_pickle('test/sample_spikes')

def test_downsample():
    x = np.random.rand(14,3)
    (x2,leftover) = simpleDownSample(x,3)
    x2_ref = x[:12,:].reshape(-1,3,3).mean(1)
    np.allclose(x2,x2_ref)
    np.allclose(leftover,x[12:,:])

    #case 1: short input
    x = np.random.rand(14,20)
    (x2,leftover) = simpleDownSample(x,20)

    assert x2 is None
    np.allclose(leftover,x)
