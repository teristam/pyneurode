from pyneurode.RingBuffer.RingBuffer import RingBuffer
import numpy as np

def test_buffer_expand():
    # test what happen when the input data is larger than the buffer
    buffer = RingBuffer((4,2))
    x = np.tile(np.arange(3), (2,1)).T
    buffer.write(x)
    
    buffer_content = np.array([[0,1,2,0], [0,1,2,0]]).T
    assert np.allclose(buffer.buffer, buffer_content)
    
    # large input
    x = np.tile(np.arange(5), (2,1)).T
    buffer.write(x)
    assert buffer.length == 9
    buffer_content = np.array([[0,1,2,0,1,2,3,4,0], [0,1,2,0,1,2,3,4,0]]).T
    assert np.allclose(buffer.buffer, buffer_content)
    
    buffer.write(x)
    buffer_content = np.array([[1,2,3,4,1,2,3,4,0], [1,2,3,4,1,2,3,4,0]]).T
    assert np.allclose(buffer.buffer, buffer_content)
