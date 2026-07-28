'''

High level idea: launch

'''
# coordination tcp store service surrounding NCCL 

class Store: 
    # insert or overwrite byte value 
    def set(self, key : str, value: bytes) -> None: 
        pass 

    # block until key exists or timeout expires 
    def get(self, key : str, timeout_s: float = None) -> bytes: 
        pass 

    # atomically increment ingeger value and return new value 
    def add(self, key, delta) -> int: 
        pass 

    # block until every requested key exists 
    def wait(self, keys: list[str], timeout_s: float = None) -> None: 
        pass 



class TCPStoreServer: 
    pass 


class TCPStoreClient: 
    pass 

