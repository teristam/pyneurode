import datetime
import logging
from multiprocessing import Value
import pickle
import time
from collections import deque
from itertools import chain, islice
from pyneurode.RingBuffer.RingBuffer import RingBuffer


class sdeque(deque):
    # A modified deque with slicing feature
    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if index.start is None or index.start>=0 else len(self)+index.start
            stop = index.stop if index.stop is None or index.stop>0 else len(self)+index.stop
            return type(self)(islice(self, start, stop, index.step))
        return deque.__getitem__(self, index)


class RingBufferG(RingBuffer):
    #A generic ring buffer that can hold any object
    def __init__(self,length):
        #shape is the shape of the ring buffer, as in the shape argument to np.zeros
        self.length = length
        self.absWriteHead = 0 #the absolute write head
        self.absReadHead = 0
        self.absRecordHead = 0
        self.buffer = sdeque(maxlen=length)
        self.isRecordingEnabled = False
        self.dataFile = None
        self.recordingBlockSize = self.length//10 #how many data to write to the disk each time
        self.logger = logging.getLogger(__class__.__name__)

    def write(self,xs):
        if isinstance(xs,list):
            for i in xs:
                self.buffer.append(i)
                self.absWriteHead += 1
        else:
            self.buffer.append(xs);
            self.absWriteHead += 1

        len2read = self.absWriteHead - self.absRecordHead

        if self.isRecordingEnabled and (len2read > self.recordingBlockSize):
            #write the data to file
            if len2read>self.length:
                #if the record head is too far behind
                x = list(self.buffer)
                self.absRecordHead = self.absWriteHead #update the record head
                self.logger.debug('recordHead too far behind. Skipping some data')
            else:
                x = list(self.buffer[-len2read:])
                self.absRecordHead += len2read

            self.write2file(x)


    def readAvailable(self):
        # read new data since last call
        length2read = self.absWriteHead-self.absReadHead

        # print(self.buffer)

        if length2read > self.length:
            self.logger.warning('Some data has already been overwritten')
            # Read the current latest data
            data = self.read(self.absWriteHead-self.length, self.length)
            self.absReadHead = self.absWriteHead
        else:
            data = self.read(self.absReadHead, length2read)
            self.absReadHead += len(data)

        return data


    def readLatest(self, length=None):

        if length is None:
            length = min(self.length, self.absWriteHead)

        #return the latest data
        if length > self.length:
            raise ValueError('Size request larger than buffer size')

        return list(self.buffer[-length:])

    def read(self,absStart,range):
        #read the data from the buffer as specified by the start index (from the very beginning)
        #and its range
        if absStart < (self.absWriteHead - self.length):
            raise ValueError('Data already overwritten')

        if absStart > self.absWriteHead:
            raise ValueError('Data too far away in the future')

        # Over wrapping to old data
        # here absWriteHead is always pointing to the end of the deque

        if self.absWriteHead < self.length:
            idx_start = 0
        else:
            idx_start = self.absWriteHead - self.length

        startPt = absStart - idx_start
        return list(self.buffer[startPt:startPt+range])

    def startRecording(self, filename=None, recordingBlockSize=None):
        if recordingBlockSize:
            self.recordingBlockSize = recordingBlockSize

        if self.recordingBlockSize > self.length:
            raise ValueError('Record block size is larger than buffer size')

        if not self.isRecordingEnabled:
            #start recording, create a data file
            if not filename:
                ts = time.time()
                filename = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')+'.pkl'
            self.dataFile = open(filename,'wb')
            self.absReadHead = self.absWriteHead #discard all data before
            self.isRecordingEnabled = True

            return filename


    def stopRecording(self):
        if self.isRecordingEnabled:
            self.isRecordingEnabled = False

            #Flush the remaining data
            segment = self.absWriteHead - self.absRecordHead
            x = self.readLatest(segment)
            self.write2file(x)

            #close file and stop recording
            self.dataFile.close()
            self.dataFile = None

    def write2file(self,x):
        #write data to file
        pickle.dump(x,self.dataFile)

    @staticmethod
    def jointParadigmArray(paradigm):
        output ={}
        for v in paradigm[0]:
            x = [p[v] for p in paradigm]
            try:
                output[v] = np.vstack(x)
            except:
                output[v] = x

        return output
    
    def readPickleStreamFile(filename):
        x = []
        segmentCount = 0
        # fastest way is to append then concatenate together
        with open(filename,'rb') as f:
            while True:
                try:
                    x.append(pickle.load(f))
                    segmentCount += 1
                    # print(f'{segmentCount}')
                except EOFError:
                    break

        print(f'Total segment read: {segmentCount}')
        
        output = list(chain(*x))
        
        return output
        # return RingBufferG.jointParadigmArray(output)




if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)

    buffer = RingBufferG(10)

    for i in range(8):
        buffer.write(i)

    print('available',buffer.readAvailable())

    for i in [8,9,10,11]:
        buffer.write(i)

    print(list(buffer.buffer))    
    assert list(buffer.buffer) == [2,3,4,5,6,7,8,9,10,11]

    assert buffer.readLatest(3) == [9,10,11]

    assert buffer.read(3, 2) == [3,4]

    print('available',buffer.readAvailable())
    assert buffer.readAvailable() == [2,3,4,5,6,7,8,9,10,11]

    for i in [12,13]:
        buffer.write(i)
    
    assert buffer.readAvailable() == [12,13]
    assert buffer.readAvailable() == []


    print(list(buffer.buffer))    

    assert buffer.read(5, 2) == [5,6]

