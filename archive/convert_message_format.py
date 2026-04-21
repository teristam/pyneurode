#%%
import pickle 
import importlib
from pyneurode.processor_node import Message

#%% Update message format
filename = 'E:\decoder_test_data\JM8_2022-09-03_16-57-23_test1\JM8_20220903_165714_9b0440_packets.pkl'
f = open(filename, 'rb')
fout = open(r'E:\decoder_test_data\JM8_2022-09-03_16-57-23_test1\test_packets.pkl','wb')
# %%
while True:
    try:
        msg = pickle.load(f)
        dtype = msg.type
        importlib.reload(Message)
        
        if type(msg) is list:
            dtype = msg.type

            new_msg = [Message.Message(dtype, m.data, m.timestamp) for m in msg]
        new_msg = Message.Message(dtype, msg.data, msg.timestamp)
        pickle.dump(new_msg, fout)
    except EOFError:
        break
# %%
