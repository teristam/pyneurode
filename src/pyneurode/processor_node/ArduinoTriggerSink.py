from pyneurode.processor_node.Processor import MsgTimeSource, Sink
from pyneurode.processor_node.Message import Message
import logging
import time
import serial

from pyneurode.processor_node.ProcessorContext import ProcessorContext

class ArduinoTriggerSink(Sink):
    """This class is intended for latency measurement. It should be used with the DigitalTrigger.ino in the repo
    it sends string with the port to trigger to arduino delimieted by space e.g. '13 12 10'
    The default pulse width is 1ms. It can be changed in the arduino sketch file
    
    Args:
        Sink (_type_): _description_
    """
    def __init__(self, port, baudrate = 115200):
        """Constructor

        Args:
            port (string): port of the Arduino device to connect to
        """
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        
    def startup(self):    
        # the Arduino object must be instantiated after process is spawned
        self.serial = serial.Serial(self.port, baudrate=self.baudrate)
        
    def process(self, message: Message):
        """Communicate with arduino via Firmata

        Args:
            message (Message): it expects a message of type 'arduino', data should be a string of format 'D<digital port> <1 or 0>', e.g.
                "D1 1" will write 1 to the digital port 1, "D2 0" will write 0 to the digital port 0
            
        """
    
        if isinstance(message, ArduinoTriggerMessage):
            # self.log(logging.DEBUG, message)
            self.serial.write(bytes(message.data.encode()))
            self.serial.readline() # don't overflow the serial port, wait for acknowledge first


class ArduinoTriggerMessage(Message):
    type = 'arduino_trigger'
    
    def __init__(self, digital_ports:list):   
        self.data = ' '.join([str(p) for p in digital_ports])+'\n'
        self.timestamp = time.time()
        
        
 

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)

    
    msg = ArduinoTriggerMessage([13,12,10])
    
    with ProcessorContext() as ctx:
        
        
        msgTimeSource = MsgTimeSource(0.001, [msg] )
        arduinoSink = ArduinoTriggerSink("COM5")
        
        msgTimeSource.connect(arduinoSink)
        
        ctx.register_processors(msgTimeSource, arduinoSink)

        ctx.start()
                    
                
            
            
        
        