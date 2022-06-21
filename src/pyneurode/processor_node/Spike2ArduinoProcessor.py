from sklearn import neural_network
from pyneurode.processor_node.Processor import MsgTimeSource, Processor
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.SpikeSortProcessor import SpikeTrainMessage
from pyneurode.processor_node.ArduinoSink import ArduinoMessage, ArduinoSink, ArduinoWaitMessage
import logging
import numpy as np

class Spike2ArduinoProcessor(Processor):
    """Convert the spike of a neuron to AdruinoMessage so that it can be used to trigger external device
    """
    
    def __init__(self, neuron_idx, digital_port, delay=0.001):
        super().__init__()
        
        """Constructor
        TODO: support multiple neurons and multiple digital outputs
        Args:
            neuron_idx (_type_): index of the neuron in the spike train to use as trigger
            delay (float, optional): delay bettween the on and off of the TTL pulse in second. Defaults to 0.001.
        """
        self.neuron_idx = neuron_idx
        self.digital_port = digital_port
        self.delay = delay
    
    def process(self, message: SpikeTrainMessage) -> SpikeTrainMessage:
        if isinstance(message, SpikeTrainMessage):
            spiketrain = message.data
            if sum(spiketrain[self.neuron_idx,:]) > 0:
                # treat it as a single event for now
                return (ArduinoMessage(self.digital_port, True), 
                        ArduinoWaitMessage(self.delay), 
                        ArduinoMessage(self.digital_port, False))

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)

    
    spike_msg1 = SpikeTrainMessage(np.array([[0,1,0,0]]).T)
    print(spike_msg1.data.shape)
    spike_msg2 = SpikeTrainMessage(np.array([[0,1,1,0]]).T)
    spike_msg3 = SpikeTrainMessage(np.array([[0,1,0,0]]).T)
    spike_msg4 = SpikeTrainMessage(np.array([[0,1,1,0]]).T)


    
    with ProcessorContext() as ctx:
        
        
        msgTimeSource = MsgTimeSource(1, [spike_msg1, spike_msg2, spike_msg3, spike_msg4] )
        spike2arduino = Spike2ArduinoProcessor(1, 13, 0.005)
        arduinoSink = ArduinoSink("COM4")
        
        msgTimeSource.connect(spike2arduino)
        spike2arduino.connect(arduinoSink)
        
        ctx.register_processors(msgTimeSource, spike2arduino, arduinoSink)

        ctx.start()
                    