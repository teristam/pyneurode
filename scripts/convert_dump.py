import io
import pickle

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


with open('data/ch256_dump.pkl','rb') as f, open('data/ch256_dump2.pkl','wb') as f2:
    while True:
        try:
            data = renamed_load(f)
            pickle.dump(data, f2)
        except EOFError:
            print('Save finished')
            break