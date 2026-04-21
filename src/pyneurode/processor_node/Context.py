from multiprocessing import get_context
import threading 

class Context:
    # to be inherit from any class that want to access the context stack
    local_storage = threading.local()
    
    @classmethod
    def get_contexts(cls):
        # get the context stack
        if not hasattr(cls.local_storage, 'contexts'):
            cls.local_storage.contexts = []

        return cls.local_storage.contexts
    
    @classmethod
    def get_context(cls):
        # return the deepest context
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError('No context in stack')
        
    def __enter__(self):
        type(self).get_contexts().append(self) # add itself to context stack
        print('Registering context to local storage')
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        # clean up
        type(self).get_contexts().pop()