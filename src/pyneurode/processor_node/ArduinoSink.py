from pyneurode.processor_node.Processor import MsgTimeSource, Sink
from pyneurode.processor_node.Message import Message
from pyfirmata import Arduino, util
import logging
import time

from pyneurode.processor_node.ProcessorContext import ProcessorContext

class ArduinoSink(Sink):
    """This class communiate with an Arduino ddevice via the Firmata protocol.
    Currently it only support writing to digital ports. For message format it expects please see the process() function

    Args:
        Sink (_type_): _description_
    """
    def __init__(self, port):
        """Constructor

        Args:
            port (string): port of the Arduino device to connect to
        """
        super().__init__()
        self.port = port
        
    def startup(self):    
        # the Arduino object must be instantiated after process is spawned
        self.board = Arduino(self.port)
        
    def process(self, message: Message):
        """Communicate with arduino via Firmata

        Args:
            message (Message): it expects a message of type 'arduino', data should be a string of format 'D<digital port> <1 or 0>', e.g.
                "D1 1" will write 1 to the digital port 1, "D2 0" will write 0 to the digital port 0
            
        """
    
        if isinstance(message, ArduinoMessage):
            s = message.data.split(" ")
            if s[0][0] == 'D':
                dport = int(s[0][1:])
                value = int(s[1])
                self.board.digital[dport].write(value)



class ArduinoMessage(Message):
    type = 'arduino'
    
    def __init__(self, digital_port, value:bool):   
        self.data =f'D{digital_port:d} {int(value)}'
        self.timestamp = time.time()

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)

    
    on_msg = ArduinoMessage(13, 1)
    off_msg = ArduinoMessage(13,0)
    
    with ProcessorContext() as ctx:
        
        
        msgTimeSource = MsgTimeSource(0.5, [on_msg, off_msg] )
        arduinoSink = ArduinoSink("COM4")
        
        msgTimeSource.connect(arduinoSink)
        
        ctx.register_processors(msgTimeSource, arduinoSink)

        ctx.start()
                    
                
            
            
        
        