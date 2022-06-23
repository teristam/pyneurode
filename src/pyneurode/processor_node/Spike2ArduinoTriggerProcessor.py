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
    
    def __init__(self, neuron_idx:list, digital_port:list):
        super().__init__()
        
        """Constructor
        TODO: support multiple neurons and multiple digital outputs
        Args:
            neuron_idx (_type_): index of the neuron in the spike train to use as trigger
            delay (float, optional): delay bettween the on and off of the TTL pulse in second. Defaults to 0.001.
        """
        assert len(neuron_idx) == len(digital_port), "The number of neuron and digital port must be the same"
        self.neuron_idx = neuron_idx
        self.digital_port = digital_port
    
    def process(self, message: SpikeTrainMessage) -> SpikeTrainMessage:
        if isinstance(message, SpikeTrainMessage):
            spiketrain = message.data
            port_list = []

            for i,idx in enumerate(self.neuron_idx):
                if idx < spiketrain.shape[0]: # do some safety check to make sure index is valid
                    if sum(spiketrain[idx,:]) > 0:
                        # collect the port to trigger
                        port_list.append(self.digital_port[i])
            
            if len(port_list)>0:
                return ArduinoTriggerMessage(port_list)

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)

    
    spike_msg1 = SpikeTrainMessage(np.array([[0,1,0,0]]).T)
    print(spike_msg1.data.shape)
    spike_msg2 = SpikeTrainMessage(np.array([[0,0,1,0]]).T)
    spike_msg3 = SpikeTrainMessage(np.array([[0,1,0,0]]).T)
    spike_msg4 = SpikeTrainMessage(np.array([[0,0,1,0]]).T)


    
    with ProcessorContext() as ctx:
        
        
        msgTimeSource = MsgTimeSource(0.001, [spike_msg1, spike_msg2, spike_msg3, spike_msg4] )
        spike2arduino = Spike2ArduinoTriggerProcessor([1,2,6,1,2,2,1],[13,12,11,10,9,8,7])
        arduinoSink = ArduinoTriggerSink("COM5")
        
        msgTimeSource.connect(spike2arduino)
        spike2arduino.connect(arduinoSink)
        
        ctx.register_processors(msgTimeSource, spike2arduino, arduinoSink)

        ctx.start()
                    