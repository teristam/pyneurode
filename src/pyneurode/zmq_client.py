
import pandas as pd 

def convert2plotdf(df):
    # convert the original spike df to one suitable for plot
    df_plot = df.set_index(['spike_id']).apply(pd.Series.explode).reset_index()
    df_plot = df_plot.apply(pd.to_numeric) #change all columns to numeric
    return df_plot


import matplotlib.pyplot as plt
import zmq
import sys
import numpy as np
import json
import uuid
import time
__author__ = 'fpbatta'


class OpenEphysEvent(object):
    event_types = {0: 'TIMESTAMP', 1: 'BUFFER_SIZE', 2: 'PARAMETER_CHANGE',
                   3: 'TTL', 4: 'SPIKE', 5: 'MESSAGE', 6: 'BINARY_MSG'}

    def __init__(self, _d, _data=None):
        self.type = None
        self.eventId = 0
        self.sampleNum = 0
        self.eventChannel = 0
        self.numBytes = 0
        self.data = b''
        self.__dict__.update(_d)
        self.timestamp = None
        # noinspection PyTypeChecker
        self.type = OpenEphysEvent.event_types[self.type]
        if _data:
            self.data = _data
            self.numBytes = len(_data)
        if self.type == 'TIMESTAMP':
            t = np.frombuffer(self.data, dtype=np.int64)
            self.timestamp = t[0]

    def set_data(self, _data):
        self.data = _data
        self.numBytes = len(_data)

    def __str__(self):
        ds = self.__dict__.copy()
        del ds['data']
        return str(ds)

    def __repr__(self):
        return self.__str__

class OpenEphysSpikeEvent(object):
    def __init__(self, _d=None, _data=None):
        self.n_channels = 0
        self.n_samples = 0
        # self.pc_proj = []
        # self.gain = []
        self.electrode_id = 0
        self.sorted_id = 0
        self.timestamp = 0
        self.channel = 0
        self.threshold = []
        # self.color = []
        self.source = 0
        self.data = None
        self.acq_timestamp = -1 #used to keep track of spike sorting time
        if _d:
            self.__dict__.update(_d) #update object data with spike event content
        # format spike data
        if type(_data) is np.ndarray:
            self.data = _data # no need converstion
        else:
            n_arr = np.frombuffer(_data, dtype=np.float32)
            n_arr = np.reshape(n_arr, (self.n_channels, self.n_samples))
            self.data = n_arr

    def __str__(self):
        ds = self.__dict__.copy()
        del ds['data']
        return str(ds)

    def _repr_pretty_(self, p, cycle):
        s = str(self)
        if self.data is not None:
            s+= f' Spike data shape:{self.data.shape}'
        else:
            s+= ' No spike data'
        p.text(s)

    

class SpikeSortClient(object):  # TODO more configuration stuff that may be obtained
    def __init__(self, sub_url="tcp://localhost:5556", req_url="tcp://localhost:5557"):
        # keep this slot for multiprocessing related initialization if needed
        self.context = zmq.Context()
        self.data_socket = None
        self.event_socket = None
        self.poller = zmq.Poller()
        self.message_no = -1
        self.socket_waits_reply = False
        self.event_no = 0
        self.app_name = 'Plot Process'
        self.uuid = str(uuid.uuid4())
        self.last_heartbeat_time = 0
        self.last_reply_time = time.time()
        self.isTesting = True
        self.data_list=[]
        self.start_time = None
        self.total_poll_time = None
        self.store_data = None
        self.sub_url = sub_url
        self.req_url = req_url

    def startup(self):
        pass

    @staticmethod
    def param_config():
        # TODO we'll have to pass the parameter requests via a second socket
        # this is meant to support a mechanism to set parameters of the application from the Open Ephys GUI.
        # not sure if it will be needed actually, it may disappear
        return ()

    def update_plot(self, n_arr):
        pass

    def update_plot_event(self, event):
        pass

    # noinspection PyMethodMayBeStatic
    def update_plot_spike(self, spike):
        print(spike)
        # pass

    def send_heartbeat(self):
        d = {'application': self.app_name, 'uuid': self.uuid, 'type': 'heartbeat'}
        j_msg = json.dumps(d)
        # print("sending heartbeat")
        self.event_socket.send(j_msg.encode('utf-8'))
        self.last_heartbeat_time = time.time()
        self.socket_waits_reply = True

    def send_event(self, event_list=None, event_type=3, sample_num=0, event_id=2, event_channel=1):
        if not self.socket_waits_reply:
            self.event_no += 1
            if event_list:
                for e in event_list:
                    self.send_event(event_type=e['event_type'], sample_num=e['sample_num'], event_id=e['event_id'],
                                    event_channel=e['event_channel'])
            else:
                de = {'type': event_type, 'sample_num': sample_num, 'event_id': event_id % 2 + 1,
                      'event_channel': event_channel}
                d = {'application': self.app_name, 'uuid': self.uuid, 'type': 'event', 'event': de}
                j_msg = json.dumps(d)
                # print(j_msg)
                if self.socket_waits_reply:
                    print("Can't send event")
                else:
                    self.event_socket.send(j_msg.encode('utf-8'), 0)
            self.socket_waits_reply = True
            self.last_reply_time = time.time()
        else:
            # pass
            print("can't send event, still waiting for previous reply")

    def callback(self, total_poll_time=None, store_data=None):
        events = []
        data_list = [] #need to use list to avoid the same message getting overwritten

        if not self.data_socket:
            print("init socket")
            self.data_socket = self.context.socket(zmq.SUB) 
            self.data_socket.connect(self.sub_url)

            self.event_socket = self.context.socket(zmq.REQ)
            self.event_socket.connect(self.req_url)

            # self.data_socket.connect("ipc://data.ipc")
            self.data_socket.setsockopt(zmq.SUBSCRIBE, b'')
            self.poller.register(self.data_socket, zmq.POLLIN)
            self.poller.register(self.event_socket, zmq.POLLIN)
            self.start_time = time.time()

        # send every two seconds a "heartbeat" so that Open Ephys knows we're alive

        if self.isTesting:
            if np.random.random() < 0.05:
                self.send_event(event_type=3, sample_num=0, event_id=self.event_no, event_channel=1)

        while True:
            #the second spike may get overwritten
            if (time.time() - self.last_heartbeat_time) > 2.:
                if self.socket_waits_reply:
                    print("heartbeat haven't got reply, retrying...")
                    self.last_heartbeat_time += 1.
                    if (time.time() - self.last_reply_time) > 10.:
                        # reconnecting the socket as per the "lazy pirate" pattern (see the ZeroMQ guide)
                        print("looks like we lost the server, trying to reconnect")
                        self.poller.unregister(self.event_socket)
                        self.event_socket.close()
                        self.event_socket = self.context.socket(zmq.REQ)
                        self.event_socket.connect("tcp://localhost:5557")
                        self.poller.register(self.event_socket)
                        self.socket_waits_reply = False
                        self.last_reply_time = time.time()
                else:
                    self.send_heartbeat()

            socks = dict(self.poller.poll(1))
            
            if not socks:
                # print("poll exits")
                break

            # Processed recieved data and event
            # mesage[1]: header
            # message[2]: data
            if self.data_socket in socks:
                data_received = {}

                try:
                    message = self.data_socket.recv_multipart(zmq.NOBLOCK)
                except zmq.ZMQError as err:
                    print("got error: {0}".format(err))
                    break
                if message:
                    if len(message) < 2:
                        print("no frames for message: ", message[0])
                    try:
                        header = json.loads(message[1].decode('utf-8'))
                    except ValueError as e:
                        print("ValueError: ", e)
                        print(message[1])

                    if self.message_no != -1 and header['message_num'] != self.message_no + 1:
                        print("missing a message at number", self.message_no)
                    
                    self.message_no = header['message_num']
                    
                    if header['type'] == 'data':
                        
                        c = header['content']
                        # print(c)
                        n_samples = c['num_samples']
                        channel_num = c['channel_num'] 
                        time_stamp = c['sample_num']

                        try:
                            # now the data from each channels are sent separately
                            n_arr = np.frombuffer(message[2], dtype=np.float32)
                            n_arr = np.reshape(n_arr, n_samples)

                            
                            data_received['type'] = 'data'
                            data_received['data'] = n_arr
                            data_received['channel_num'] = channel_num
                            data_received['data_timestamp'] = time_stamp
                            data_list.append(data_received)
                            # data = {
                            #     'data':n_arr,
                            #     'channel_num': channel_num,
                            #     'data_timestamp': time_stamp
                            # }
                            
                            # data_received['data'].append(data)

                        except IndexError as e:
                            print(e)
                            print(header)
                            print(message[1])
                            if len(message) > 2:
                                print(len(message[2]))
                            else:
                                print("only one frame???")

                    # elif header['type'] == 'event':

                    #     if header['data_size'] > 0:
                    #         event = OpenEphysEvent(header['content'], message[2])
                    #     else:
                    #         event = OpenEphysEvent(header['content'])
                    #     self.update_plot_event(event)

                    elif header['type'] == 'spike':
                        spike = OpenEphysSpikeEvent(header['spike'], message[2])
                        # print(spike.data)
                        #Extract spike data
                        data_received['type'] = 'spike'
                        data_received['spike'] = spike
                        data_list.append(data_received)
                        # self.update_plot_spike(spike)

                    # elif header['type'] == 'param':
                    #     c = header['content']
                    #     self.__dict__.update(c)
                    #     print(c)
                    # else:
                    #     raise ValueError("message type unknown")
                    
        
                        
                    
                else:
                    print("got no data")

                    break
            elif self.event_socket in socks and self.socket_waits_reply:
                # Evetn reply received
                message = self.event_socket.recv()
                # print("event reply received")
                # print(message)
                if self.socket_waits_reply:
                    self.socket_waits_reply = False

                else:
                    print("???? getting a reply before a send?")

            # data_list.append(data_received)
        # print "finishing callback"
        # if store_data:
        #     self.data_list.append(data_received)
        if events:
            pass  # TODO implement the event passing
        
        
        data = combine_data_channel(data_list, 
                                            list(range(27)))

        if data is not None:
            print(data['data'].shape)
            
        return data_list

    @staticmethod
    def terminate():
        plt.close()
        sys.exit(0)

def combine_data_channel(data_list, channel_list:list):
    
    # if len(data_list) >= len(channel_list):
    data = [None] * len(channel_list)
    data_all = []
    sample_idx = None
    start_timestamp = 0
    
    # combine data from all channel into one array
    for d in data_list:
        if d['type'] == 'data':
            try:
                idx = channel_list.index(d['channel_num'])
                # print('channel list idx', idx)
                
                # make sure the data are from the same timestamp
                if sample_idx is None:
                    sample_idx = d['data_timestamp']
                    start_timestamp = sample_idx
                else:
                    if not sample_idx == d['data_timestamp']:
                        # new data segment
                        data_all.append(np.stack(data));
                        sample_idx = d['data_timestamp']
                        
                        # create a new holder
                        data = [None] * len(channel_list)
                                            
                data[idx] = d['data']
     
            except ValueError:
                    print('Channel number not found in data list')
                
        # make sure we get all the data
        # assert len(data) == len(channel_list), f'Number of channel does not match {len(data)} expected: {len(channel_list)}'
        # print(channel_list, data)
        
        # make sure all data are aligned to the same timestamp
        # assert np.all(sample_idx == sample_idx[0]), f'Data not aligned to the same timestamp {sample_idx}'
        
    if len(data_all)>0:
        data = {}
        data['type'] = 'data'
        data['data'] = np.concatenate(data_all, axis=1)
        data['data_timestamp'] = start_timestamp
        return data

    
    return None