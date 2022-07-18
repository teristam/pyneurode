from pyneurode.processor_node.Processor import *
from pyneurode.processor_node.ProcessorContext import ProcessorContext
from pyneurode.processor_node.FileReaderSource import FileReaderSource
from pyneurode.processor_node.SpikeSortProcessor import SpikeSortProcessor
from pyneurode.processor_node.SyncDataProcessor import SyncDataProcessor
import pyneurode as dc
import logging

logger = logging.getLogger(__name__)


def test_processor():
    start  = time.time()
    echoFile =  FileEchoSink.get_temp_filename()

    print(echoFile)

    with ProcessorContext() as ctx:
        
        source = DummyTimeSource(0.1)
        add = AddProcessor()
        echo1 = FileEchoSink(filename=echoFile)
        echo2 = EchoSink()

        source.connect(add)
        add.connect(echo1)
        add.connect(echo2)

        ctx.register_processors(source, add, echo1, echo2)

        ctx.start()

        time.sleep(3)
        logging.debug('Sending the stop signal now')
        # send stop signal after 5s
        ctx.shutdown_event.set()

    time.sleep(1)    
    data = FileEchoSink.open_data_dump(echoFile)
    assert len(data) > 10


def test_filereadersource():
    start  = time.time()
    echofile = FileEchoSink.get_temp_filename()

    with ProcessorContext() as ctx:

        fileReader = FileReaderSource('data/data_packets.pkl', interval=0.01)
        echo = FileEchoSink(echofile)
        fileReader.connect(echo,'adc_data')

        ctx.register_processors(fileReader, echo)
        ctx.start()

        time.sleep(5)
        logger.debug('Sending the stop signal now')
        ctx.shutdown_event.set()

    time.sleep(1)  
    data = FileEchoSink.open_data_dump(echofile)
    print(len(data))
    assert len(data) > 5


def test_spikesortprocessor():
    start  = time.time()
    echofile = FileEchoSink.get_temp_filename()

    with ProcessorContext() as ctx:
        fileReader = FileReaderSource('data/data_packets_M2_D23_1910.pkl')
        spikeSortProcessor = SpikeSortProcessor(interval=0.02)
        fileEchoSink = FileEchoSink(echofile)

        fileReader.connect(spikeSortProcessor, filters='spike')
        spikeSortProcessor.connect(fileEchoSink, filters='spike_train')

        ctx.register_processors(fileReader, spikeSortProcessor, fileEchoSink)

        ctx.start()

        time.sleep(20)
        logger.debug('Sending the stop signal now')
        ctx.shutdown_event.set()


    time.sleep(1)  
    data = FileEchoSink.open_data_dump(echofile)
    assert len(data) > 1


def test_syncdataprocessor():
    start  = time.time()
    echofile = FileEchoSink.get_temp_filename()

    with ProcessorContext() as ctx:

        fileReader = FileReaderSource('data/data_packets_M2_D23_1910.pkl')
        spikeSortProcessor = SpikeSortProcessor(interval=0.02)
        syncDataProcessor = SyncDataProcessor(interval=0.02)
        fileEchoSink = FileEchoSink(echofile)

        fileReader.connect(spikeSortProcessor, filters='spike')
        spikeSortProcessor.connect(syncDataProcessor)
        fileReader.connect(syncDataProcessor, 'adc_data')
        syncDataProcessor.connect(fileEchoSink)

        ctx.register_processors(fileReader, spikeSortProcessor, syncDataProcessor, fileEchoSink)

        ctx.start()

        time.sleep(20)
        ctx.shutdown_event.set()

    time.sleep(1)  
    data = FileEchoSink.open_data_dump(echofile)

    assert len(data)> 10

def test_fileEchoSource():
    start  = time.time()
    echofile = 'test/data/syncfile.pkl'

    with ProcessorContext() as ctx:

        fileEcho = FileEchoSource(interval=0.1, filename=echofile)
        echo = EchoProcessor()
        fileEcho.connect(echo)

        ctx.register_processors(fileEcho, echo)
        ctx.start()

        time.sleep(20)
        logger.debug('Sending the stop signal now')
        ctx.shutdown_event.set()

    