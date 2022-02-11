from pickle import TRUE
import pytest
from pyneurode.RingBuffer.RingBuffer import RingBuffer
import numpy  as np 

@pytest.fixture
def ringbuffer():
    return RingBuffer((3,2))

@pytest.fixture
def ringbuffer_written():
    # ring buffer always has content written
    # the content insde the buffer is [[3,6],[1,2],[2,4]]
    ringbuffer = RingBuffer((3,2))
    for i in range(4):
        ringbuffer.write(i*np.array([[1,2]]))

    return ringbuffer

@pytest.fixture
def ringbuffer_timestamp():
    ringbuffer = RingBuffer((3,2),start_timestamp=3, dt=0.1)
    for i in range(4):
        ringbuffer.write(i*np.array([[1,2]]))
    return ringbuffer

def test_write():
    ringbuffer=RingBuffer((3,2))
    for i in range(4):
        ringbuffer.write(i*np.array([[1,2]]))
    
    expected = np.array([[3,1,2],[6,2,4]]).T
    assert np.allclose(ringbuffer.buffer,expected)

def test_readLatest(ringbuffer_written):
    x = ringbuffer_written.readLatest(2)

    expected = np.array([[2,4],[3,6]])
    assert np.allclose(x,expected)

    x = ringbuffer_written.readLatest()
    expected = np.array([[1,2],[2,4],[3,6]])
    assert np.allclose(x,expected)

def test_read(ringbuffer_written:RingBuffer):
    x = ringbuffer_written.read(1,3)
    expected = np.array([[1,2],[2,4],[3,6]])
    assert np.allclose(x,expected)

def test_timestamp_read(ringbuffer_timestamp:RingBuffer):
    x,ts = ringbuffer_timestamp.readLatest(2,return_timestamps=True)
    assert np.allclose(ts, np.array([3.2,3.3]))
    
