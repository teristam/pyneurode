from sklearn import neural_network
from pyneurode.processor_node.Processor import MsgTimeSource, Processor
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.SpikeSortProcessor import SpikeTrainMessage
from pyneurode.processor_node.ArduinoTriggerSink import ArduinoTriggerMessage, ArduinoTriggerSink
import logging
import numpy as np

class Spike2ArduinoTriggerProcessor(Processor):
    """Convert the spike of a neuron to AdruinoMessage so that it can be used to trigger external device
    """
    
    def __init__(self, neuron_idx:list, digital_port:list, max_frame_to_skip=100):
        super().__init__()
        
        """Constructor
        TODO: support multiple neurons and multiple digital outputs
        Args:
            neuron_idx (_type_): index of the neuron in the spike train to use as trigger
            delay (float, optional): delay bettween the on and off of the TTL pulse in second. Defaults to 0.001.
            max_frame_to_skip(int,optional): max length of spiktrain above witch the spike event will be omit for performance reason. Defaults to 100
        """
        assert len(neuron_idx) == len(digital_port), "The number of neuron and digital port must be the same"
        self.neuron_idx = neuron_idx
        self.digital_port = digital_port
        self.max_frame_to_skip = max_frame_to_skip
    
    def process(self, message: SpikeTrainMessage) -> SpikeTrainMessage:
        if isinstance(message, SpikeTrainMessage):
            spiketrain = message.data

            if spiketrain.shape[1]<self.max_frame_to_skip:
                msg_list = []
                for i in range(spiketrain.shape[1]):
                    #send pulse for every bin in the spiketrain
                    port_list = []
                    for idx, j in enumerate(self.neuron_idx):
                        if j < spiketrain.shape[0]:
                            if spiketrain[j,i] >0: # check to see if there is any spikes
                                port_list.append(self.digital_port[idx]) # collect the port to trigger
                        
                    if len(port_list)>0:
                        msg_list.append(ArduinoTriggerMessage(port_list))
            
                if len(msg_list)>0:
                    return msg_list

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)

    
    spike_msg1 = SpikeTrainMessage(np.array([[0,1,0,0], [1,0,0,0],[0,1,0,0]]).T)
    print(spike_msg1.data.shape)
    spike_msg2 = SpikeTrainMessage(np.array([[0,0,1,0]]).T)
    spike_msg3 = SpikeTrainMessage(np.array([[0,1,0,0]]).T)
    spike_msg4 = SpikeTrainMessage(np.array([[0,0,1,0]]).T)


    
    with ProcessorContext() as ctx:
        
        
        msgTimeSource = MsgTimeSource(0.001, [spike_msg1, spike_msg2, spike_msg3, spike_msg4] )
        spike2arduino = Spike2ArduinoTriggerProcessor([1],[13])
        arduinoSink = ArduinoTriggerSink("COM4")
        
        msgTimeSource.connect(spike2arduino)
        spike2arduino.connect(arduinoSink)
        
        ctx.register_processors(msgTimeSource, spike2arduino, arduinoSink)

        ctx.start()
                    