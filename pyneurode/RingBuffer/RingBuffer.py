from queue import Empty
import numpy as np
import time
import datetime
import pickle
from collections import deque
from itertools import islice, chain
import logging
# import niVis.Leap as Leap
import ctypes, struct
from ctypes import addressof
import multiprocessing


class NotEnoughDataError(Exception):
    pass

class RingBuffer:
    # A ring buffer to hold numpy array
    
    def __init__(self,shape,dtype=float,dt=1,start_timestamp=0):
        #shape is the shape of the ring buffer, as in the shape argument to np.zeros
        # the shape is assume to be time x dimension
        self.length = shape[0]
        self.absWriteHead = 0 #the absolute write head
        self.absReadHead = 0
        self.absRecordHead = 0
        self.buffer = np.zeros(shape,dtype=dtype)
        self.timestamps = np.zeros((self.length,)) #time stamp of the data, will be tracked automatically
        self.timestamps[0] = start_timestamp
        self.dt = dt #dt of each data point
        self.start_timestamp = start_timestamp
        self.latest_time = self.start_timestamp
        self.isRecordingEnabled = False
        self.dataFile = None
        self.recordingBlockSize = self.length//10 #how many data to write to the disk each time
        self.dtype = dtype

    def set_buffer_size(self,shape,dtype=float,dt=1,start_timestamp = 0):
        # reset the buffer size
        self.buffer = np.zeros(shape,dtype=dtype)
        self.start_timestamp = start_timestamp
        self.timestamp = np.zeros((self.length,))
        self.dt = dt 
        self.length = shape[0]
        self.absWriteHead = 0 #the absolute write head
        self.absReadHead = 0
        self.absRecordHead = 0
        self.isRecordingEnabled = False
        self.dataFile = None
        self.recordingBlockSize = self.length//10 #how many data to write to the disk each time
        self.dtype = dtype


    def writeHead(self):
        return self.absWriteHead % self.length

    def write(self,xs):
        # Write the array to the Ring buffer
        # The format should be in (time x channel)
        # Portion of the array will wrap back to the beginning if it exceeds the buffer length
        if self.dtype==bytes:
            xsLength = len(xs)
        else:
            xsLength = xs.shape[0]
            if self.buffer.shape[1:] != xs.shape[1:] :
                raise ValueError('Shape mismatch')

        # TODO: may need to so things differently for bytes data
        ts_segment = np.linspace(0, xs.shape[0]*self.dt, xs.shape[0])
        ts_segment += self.latest_time
        self.latest_time += xs.shape[0]*self.dt

        if (self.writeHead()+xsLength > self.length):
            # need to wrap around to the start

            data2write = self.length - self.writeHead()
            remainData = xsLength - data2write
            if self.dtype == bytes:
                self.buffer[self.writeHead():,:] = self.bytes2numpy(xs[:data2write])
                self.buffer[:remainData,:] = self.bytes2numpy(xs[data2write:])
            else:
                self.buffer[self.writeHead():,:] = xs[:data2write,:]
                self.buffer[:remainData,:] = xs[data2write:,:]

                #keep track of time
                # self.timestamps[self.writeHead:] = ts_segment[:data2write]
                # self.timestamps[:remainData] = ts_segment[data2write:]
        else:
            if self.dtype == bytes:
                self.buffer[self.writeHead():self.writeHead()+xsLength,:] = self.bytes2numpy(xs)
            else:
                self.buffer[self.writeHead():self.writeHead()+xsLength,:] = xs
                self.timestamps[self.writeHead():self.writeHead()+xsLength] = ts_segment

        self.absWriteHead += xsLength

        len2read = self.absWriteHead - self.absRecordHead

        if self.isRecordingEnabled and (len2read > self.recordingBlockSize):
            #write the data to file
            if len2read>self.length:
                #if the record head is too far behind
                x = self.readLatest(self.length)
                self.absRecordHead = self.absWriteHead #update the record head
            else:
                x = self.readLatest(len2read)
                self.absRecordHead += len2read

            self.write2file(x)

    def readContinous(self,length,wait=False,timeoutCount=np.inf, verbose=False):
        #Read continous data from the beginning
        count =0
        if wait:
            #wait till data is ready
            while((self.absWriteHead-self.absReadHead)<length):
                if verbose:
                    print(f'write: {self.absWriteHead} read: {self.absReadHead}')
                time.sleep(0.01)
                count +=1
                if count>timeoutCount:
                    raise RuntimeError('Buffer read timeout!')
        else:
            if (self.absWriteHead-self.absReadHead)<length:
                raise NotEnoughDataError
        
        x = self.read(self.absReadHead, length)
        self.absReadHead+=length
        return x

    def readLatest(self, length=None, return_timestamps=False, verbose=False):

        if length is None:
            length = min(self.absWriteHead,self.length)

        #return the latest data
        if length > self.length:
            raise ValueError('Size request larger than buffer size')

        idx = (self.writeHead()- np.arange(length,0,-1)) % self.length #find index relative to the start of buffer

        if verbose:
            print(f'writeHead: {self.writeHead()} idx: {idx}')

        if not return_timestamps:
            return self.buffer[idx,:]
        else:
            return (self.buffer[idx,:], self.timestamps[idx])

    def bytes2numpy(self, b):
        return np.expand_dims(np.frombuffer(b,dtype='S1', count=len(b)),1)

    def getWrappedData(self,x, localStart,length2read):
        #Get data from x, if the length2read is longer than the array length, wrap around to the beginning
        # x should be in the format (time, channel)
        length = x.shape[0]
        chunk1 = x[localStart:,:]
        chunk2 = x[:(length2read - (length-localStart)),:]
        chunk = np.vstack([chunk1,chunk2])
        assert chunk.shape[0]==length2read
        return chunk


    def read(self,absStart,range,return_timstamps=False):
        #read the data from the buffer as specified by the start index (from the very beginning)
        #and its range
        if absStart < (self.absWriteHead - self.length):
            raise ValueError('Data already overwritten')

        # Over wrapping to old data
        length2read = min(range,self.absWriteHead-absStart)

        # idx = np.arange(absStart,absStart+length2read)
        # return self.buffer[idx % self.length,:]

        # use a more efficient read
        localStart = absStart % self.length
        if (localStart + length2read) > self.length:
            chunk1 = self.buffer[localStart:,:]
            chunk2 = self.buffer[:(length2read - (self.length-localStart)),:]
            chunk = np.vstack([chunk1,chunk2])
            assert chunk.shape[0]==length2read

            if not return_timstamps:
                return chunk
            else:
                #also assemble the timestamp
                ts= self.getWrappedData(self.timestamps,localStart,length2read)
                return (chunk,ts)

        else:
            if not return_timstamps:
                return self.buffer[localStart:localStart+length2read,:]
            else:
                ts = self.timestamps[localStart:localStart+length2read]
                return (self.buffer[localStart:localStart+length2read,:], ts)

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
            self.absRecordHead = self.absWriteHead 
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
        return np.vstack(x)

