from pyneurode.RingBuffer.RingBufferG import *


def test_readAvaiable():
    buffer = RingBufferG(5)

    for i in range(3):
        buffer.write(i)

    assert buffer.readAvailable() == [0,1,2]

    for i in [3,4,5]:
        buffer.write(i)

    assert list(buffer.buffer) == [1,2,3,4,5]

    assert buffer.readAvailable() == [3,4,5]


def test_readAvailableOverflow():
    buffer = RingBufferG(5)

    for i in range(7):
        buffer.write(i)

    assert buffer.readAvailable() == [2,3,4,5,6]