from pyneurode.processor_node.Processor import *
from pyneurode.RingBuffer.RingBuffer import *
from pyneurode.processor_node.ProcessorContext import *
import numpy as np 
from scipy import signal
from sklearn import svm, preprocessing
from pyneurode.utils import *
import logging


def step_decode(agg_buffer, predict_buffer, slidWin, spk_scaler, 
    pos_scaler,regressor,target_d,alpha_smooth=0.8):
 
    # get the spike rate and the true position
    x = agg_buffer.readLatest(slidWin)
    spk_train = x[:,:-1]
    pos = x[slidWin-2:,-1] #get current and previous location to detect teleport
    


    #normalize and predict
    norm_spk_train = spk_scaler.transform(spk_train)
    norm_pos = pos_scaler.transform(pos.reshape(-1,1))
    cur_pos = norm_pos[1,0]
    prev_pos = norm_pos[0,0]

    norm_spk_train = norm_spk_train.reshape(1,-1)
    pos_predicted = regressor.predict(norm_spk_train)[0] #output is an array of position


    # Determine what to do based on the current location
    # only decode in the outbound zone because ramp cell may reset at reward zone
    # TODO: different decoder in the outbound and inbound zone
    # reset when track teleport is detected
    # in the homebound zone, use the actual movement data to move forward

    prev_data = predict_buffer.readLatest(1) #the predict buffer contains both the true and predicted pos
    prev_predicted_pos = prev_data[-1,1]

    if (cur_pos-prev_pos) > 100:
        # Teleport, reset to beginning
        p_next = 0
    elif 10 < cur_pos < 120:
        # only decode in the outbound zone
        d = pos_predicted - prev_predicted_pos
        if d > 0:
            # use exponential smoothing
            target_d = prev_predicted_pos + d
            # print(target_d)
            p_next = prev_predicted_pos*(1-alpha_smooth) + alpha_smooth*target_d
        else:
            if prev_predicted_pos < target_d:
                # continue to move if we have not reached the target
                p_next = prev_predicted_pos*(1-alpha_smooth) + alpha_smooth*target_d
            else:
                # keep the same location if the predict location is not further away
                p_next = prev_predicted_pos
    else:
        # homebound region is controlled by the actual movement
        p_next = cur_pos

    # need to avoid over pulling, because the time constant of the exponential smoothing depends on the 
    # sampling freuqency

    time.sleep(0.08) # TDOO: avoid hard-coding
    return p_next, target_d


def smooth_decode(agg_buffer, predict_buffer, slidWin, spk_scaler, pos_scaler,regressor, smooth_alpha=0.8):
    # Smoothe the output by expoential smoothing
    # decode the latest data

    x = agg_buffer.readLatest(slidWin)
    spk_train = x[:,:-1]
    pos = x[slidWin-1:,-1]

    #normalize and predict
    norm_spk_train = spk_scaler.transform(spk_train)
    norm_pos = pos_scaler.transform(pos.reshape(-1,1)).ravel()

    #make sliding window
    norm_spk_train = norm_spk_train.reshape(1,-1)
    
    pos_predicted = regressor.predict(norm_spk_train)

    # Use expoential smoothing tosmooth the data
    if predict_buffer.absWriteHead > 1:
        prev_data = predict_buffer.readLatest(1) #the predict buffer contains both the true and predicted pos
        prev_predicted_pos = prev_data[-1,1]
        #do not apply smoothing at track teleport
        if (prev_predicted_pos-pos_predicted) <100:
            pos_predicted = pos_predicted*smooth_alpha+prev_predicted_pos*(1-smooth_alpha)
    
    return pos_predicted

class DecodeProcessor(Processor):
    buffer_length = 1000
    train_data_length = 100 # how much data to train the decoder
    slideWin = 5
    alpha_smooth = 0.2

    def __init__(self, proc_name: str, shutdown_event: Event, smooth_method='smooth'):
        super().__init__(proc_name, shutdown_event)
        self.smooth_method = smooth_method

    def startup(self):
        super().startup()
        self.ring_buffer = None
        self.predict_buffer = RingBuffer((self.buffer_length, 2))
        self.regressor = None
        self.pos_scaler = None
        self.spk_scaler = None
        self.decoder_ready = False
        self.target_d = 0 # target location
        

    def process(self, message: Message) -> Message:
        data = message.data
        assert isinstance(data,np.ndarray), 'Error, data must be a an array'

        if self.ring_buffer is not None:
           self.ring_buffer.write(data)
        else:
            self.ring_buffer = RingBuffer((self.buffer_length,data.shape[1]))
            self.ring_buffer.write(data)


        if self.ring_buffer is not None:
            if self.regressor is None and self.ring_buffer.absWriteHead > self.train_data_length:
                #############
                # Training
                startTime = time.perf_counter()

                agg_data = self.ring_buffer.readContinous(self.train_data_length)
            

                #Read data from buffer
                spk_train = agg_data[:,:-1] #the last channel is the position data
                pos = agg_data[:,-1]

        
                #normalize data and transform data
                scaler = preprocessing.StandardScaler()
                scaler.fit(spk_train)
                norm_spk_train = scaler.transform(spk_train)

                # make sliding feature
                norm_spk_train = makeSlidingWinFeature(norm_spk_train, self.slideWin)
                pos = pos[self.slideWin-1:]

                # scale the position data such that the output is in correct location
                pos_scaler = preprocessing.MinMaxScaler(feature_range=(0,200))
                pos_scaler.fit(pos.reshape(-1,1))
                norm_pos = pos_scaler.transform(pos.reshape(-1,1)).ravel()

                polyFeatures = preprocessing.PolynomialFeatures(1).fit(norm_spk_train)
                poly_spiketrain = polyFeatures.transform(norm_spk_train)

                # train and predict
                regressor = svm.SVR(kernel='rbf',C=50,gamma='auto')
                regressor.fit(norm_spk_train, norm_pos)

                pos_predicted = regressor.predict(norm_spk_train)

                # Smooth the output
                predicted_smooth = signal.savgol_filter(pos_predicted.ravel(),5,1)

                #Evaluate performance
                varExplain = model_selection.cross_val_score(regressor,norm_spk_train,norm_pos,cv=5)
                print(f'Training took {(time.perf_counter()-startTime)*1000:.2f}ms varExplain: {varExplain.mean()}' )

                self.decoderReady = True
                self.regressor = regressor
                self.pos_scaler = pos_scaler
                self.spk_scaler = scaler
            
            if self.regressor is not None:
                ###############
                # Inference
                agg_data = self.ring_buffer.readLatest()
                
                spk_train = agg_data[:,:-1] # the last channel is the position data
                pos_true = agg_data[:,-1]
                pos_true = self.pos_scaler.transform(pos_true.reshape(-1,1)).ravel()

                if self.smooth_method == 'step':
                    pos_predicted, self.target_d = step_decode(self.ring_buffer, self.predict_buffer, 
                        self.slideWin, self.spk_scaler, self.pos_scaler, self.regressor, self.target_d, alpha_smooth=0.2)
                else:
                    pos_predicted = smooth_decode(self.ring_buffer, self.predict_buffer, 
                        self.slideWin, self.spk_scaler, self.pos_scaler, self.regressor, self.alpha_smooth)

                pos2write = np.array([pos_true[-1], pos_predicted[-1]])
                self.predict_buffer.write(pos2write.reshape(-1,2))
                
                return Message('predicted_pos', pos2write)




# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#     start  = time.time()
#     echofile = FileEchoSink.get_temp_filename()
#     print(echofile)

#     with ProcessorContext() as ctx:
#         ctx.makeProcessor('DummySource', DummpyNumpyTimeSource, interval=0.2, shape=(10,5))
#         ctx.makeProcessor('Decode', DecodeProcessor)
#         ctx.makeProcessor('Echo', FileEchoSink, filename=echofile)

#         ctx.connect('DummySource', 'Decode')
#         ctx.connect('Decode', 'Echo')

#         ctx.start()

#         time.sleep(20)
#         logging.debug('Sending the stop signal now')
#         ctx.shutdown_event.set()

#     time.sleep(1)  
#     data = FileEchoSink.open_data_dump(echofile)
#     print(len(data))