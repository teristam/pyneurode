import io
import pickle
from pyneurode.processor_node.Message import Message
from pathlib import Path

'''
Adapted from https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
'''

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "decode_client.processor_node.Processor":
            renamed_module = "pyneurode.processor_node.Processor"
        elif module == 'zmq_client':
            renamed_module = "pyneurode.zmq_client"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


def migrate_message(obj):
    """Convert old Message objects that used 'type' attribute to the new 'dtype' attribute."""
    items = obj if isinstance(obj, list) else [obj]
    for m in items:
        if isinstance(m, Message) and 'dtype' not in m.__dict__ and 'type' in m.__dict__:
            m.dtype = m.__dict__.pop('type')
    return obj


file2convert = Path('data/M7_2022-07-16_17-17-33_test1/M7_test1_20220716_171720_305e71_packets.pkl')
with open(file2convert,'rb') as f, open(file2convert.parent/'packets_new.pkl','wb') as f2:
    while True:
        try:
            data = renamed_load(f)
            migrate_message(data)
            pickle.dump(data, f2)
        except EOFError:
            print('Save finished')
            break